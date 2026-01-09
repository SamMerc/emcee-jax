from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import device_get, random
from jax.tree_util import tree_flatten, tree_map

from emcee_jax._src.ensemble import Ensemble
from emcee_jax._src.log_prob_fn import WrappedLogProbFn, LogProbFn, wrap_log_prob_fn
from emcee_jax._src.moves.core import Extras, Move, MoveState, Stretch
from emcee_jax._src.types import Array, SampleStats

if TYPE_CHECKING:
    from arviz import InferenceData


class SamplerState(NamedTuple):
    move_state: MoveState
    ensemble: Ensemble
    extras: Extras


class Trace(NamedTuple):
    final_state: SamplerState
    samples: Ensemble
    extras: Extras
    move_state: MoveState
    sample_stats: SampleStats

    def to_inference_data(self, **kwargs: Any) -> "InferenceData":
        from arviz import InferenceData, dict_to_dataset

        import emcee_jax

        # Deal with different possible PyTree shapes
        samples = self.samples.coordinates
        if not isinstance(samples, dict):
            flat, _ = tree_flatten(samples)
            samples = {f"param_{n}": x for n, x in enumerate(flat)}

        # Deterministics also live in samples
        deterministics = self.samples.deterministics
        if deterministics is not None:
            if not isinstance(deterministics, dict):
                flat, _ = tree_flatten(deterministics)
                deterministics = {f"det_{n}": x for n, x in enumerate(flat)}
            for k in list(deterministics.keys()):
                if k in samples:
                    assert f"{k}_det" not in samples
                    deterministics[f"{k}_det"] = deterministics.pop(k)
            samples = dict(samples, **deterministics)

        # ArviZ has a different convention about axis locations. It wants (chain,
        # draw, ...) whereas we produce (draw, chain, ...).
        samples = tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)

        # Convert sample stats to ArviZ's preferred style
        sample_stats = dict(
            _flatten_dict(self.sample_stats), lp=self.samples.log_probability
        )
        renames = [("accept_prob", "acceptance_rate")]
        for old, new in renames:
            if old in sample_stats:
                sample_stats[new] = sample_stats.pop(old)
        sample_stats = tree_map(lambda x: jnp.swapaxes(x, 0, 1), sample_stats)

        return InferenceData(
            posterior=dict_to_dataset(device_get(samples), library=emcee_jax),
            sample_stats=dict_to_dataset(
                device_get(sample_stats), library=emcee_jax
            ),
            **kwargs,
        )


def _flatten_dict(
    obj: Union[Dict[str, Any], Any]
) -> Union[Dict[str, Any], Any]:
    if not isinstance(obj, dict):
        return obj
    result = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            for k1, v1 in _flatten_dict(v).items():
                result[f"{k}:{k1}"] = v1
        else:
            result[k] = v
    return result


@dataclass(frozen=True, init=False)
class EnsembleSampler:
    wrapped_log_prob_fn: WrappedLogProbFn
    move: Move
    use_pmap: bool

    def __init__(
        self,
        log_prob_fn: LogProbFn,
        *,
        move: Optional[Move] = None,
        log_prob_args: Tuple[Any, ...] = (),
        log_prob_kwargs: Optional[Dict[str, Any]] = None,
        use_pmap: bool = False,
    ):
        """Initialize the Ensemble Sampler.
        
        Args:
            log_prob_fn: The log probability function
            move: The MCMC move to use (default: Stretch)
            log_prob_args: Additional positional arguments for log_prob_fn
            log_prob_kwargs: Additional keyword arguments for log_prob_fn
            use_pmap: If True, use pmap for parallel execution across devices.
                     Requires that num_walkers is divisible by num_devices.
        """
        log_prob_kwargs = {} if log_prob_kwargs is None else log_prob_kwargs
        wrapped_log_prob_fn = wrap_log_prob_fn(
            log_prob_fn, *log_prob_args, **log_prob_kwargs
        )
        object.__setattr__(self, "wrapped_log_prob_fn", wrapped_log_prob_fn)

        move = Stretch() if move is None else move
        object.__setattr__(self, "move", move)
        object.__setattr__(self, "use_pmap", use_pmap)

    def init(
        self,
        random_key: jax.Array,
        ensemble: Union[Ensemble, Array],
    ) -> SamplerState:
        initial_ensemble = Ensemble.init(self.wrapped_log_prob_fn, ensemble)
        move_state, extras = self.move.init(random_key, initial_ensemble)
        return SamplerState(move_state, initial_ensemble, extras)

    def step(
        self,
        random_key: jax.Array,
        state: SamplerState,
        *,
        tune: bool = False,
    ) -> Tuple[SamplerState, SampleStats]:
        if not isinstance(state, SamplerState):
            raise ValueError(
                "Invalid input state; you must call sampler.init(...) "
                "to initialize the state first"
            )
        new_state, stats = self.move.step(
            self.wrapped_log_prob_fn, random_key, *state, tune=tune
        )
        return SamplerState(*new_state), stats

    def sample(
        self,
        random_key: jax.Array,
        state: SamplerState,
        num_steps: int,
        *,
        tune: bool = False,
        progress: bool = True,
        progress_desc: str = "Sampling",
    ) -> Trace:
        """Sample from the ensemble.
        
        Args:
            random_key: JAX random key
            state: Initial sampler state
            num_steps: Number of MCMC steps
            tune: Whether this is a tuning phase
            progress: Whether to show tqdm progress bar
            progress_desc: Description for progress bar
            
        Returns:
            Trace containing samples and statistics
        """
        def one_step(
            state: SamplerState, key: jax.Array
        ) -> Tuple[SamplerState, Tuple[SamplerState, SampleStats]]:
            state, stats = self.step(key, state, tune=tune)
            return state, (state, stats)

        keys = random.split(random_key, num_steps)
        if progress:
            try:
                from tqdm.auto import tqdm
                
                # Compile the step function first
                compiled_step = jax.jit(one_step)
                
                # Run with progress bar
                final_state = state
                all_states = []
                all_stats = []
                
                with tqdm(total=num_steps, desc=progress_desc) as pbar:
                    for i in range(num_steps):
                        final_state, (trace_state, stats) = compiled_step(
                            final_state, keys[i]
                        )
                        all_states.append(trace_state)
                        all_stats.append(stats)
                        pbar.update(1)
                        
                        # Optional: Update progress bar with acceptance rate
                        if "accept_prob" in stats:
                            accept = float(device_get(stats["accept_prob"]))
                            pbar.set_postfix({"accept": f"{accept:.3f}"})
                
                # Stack the results
                trace = tree_map(lambda *xs: jnp.stack(xs), *all_states)
                sample_stats = tree_map(lambda *xs: jnp.stack(xs), *all_stats)
                
            except ImportError:
                print("tqdm not installed. Running without progress bar.")
                print("Install with: pip install tqdm")
                final_state, (trace, sample_stats) = jax.lax.scan(one_step, state, keys)
        else:
            # Run without progress bar
            final_state, (trace, sample_stats) = jax.lax.scan(one_step, state, keys)
        
        return Trace(
            final_state=final_state,
            samples=trace.ensemble,
            extras=trace.extras,
            move_state=trace.move_state,
            sample_stats=sample_stats,
        )

    def sample_parallel(
        self,
        random_key: jax.Array,
        state: SamplerState,
        num_steps: int,
        *,
        tune: bool = False,
        progress: bool = True,
        progress_desc: str = "Sampling (parallel)",
    ) -> Trace:
        """Sample from the ensemble using pmap for parallelization.
        
        This method splits walkers across available devices for parallel execution.
        Requires that num_walkers is divisible by the number of devices.
        
        Args:
            random_key: JAX random key
            state: Initial sampler state
            num_steps: Number of MCMC steps
            tune: Whether this is a tuning phase
            progress: Whether to show tqdm progress bar
            progress_desc: Description for progress bar
            
        Returns:
            Trace containing samples and statistics
        """
        
        num_devices = jax.device_count()
        
        # Reshape ensemble to split across devices
        # ensemble shape: (num_walkers, ...) -> (num_devices, walkers_per_device, ...)
        def reshape_for_pmap(x):
            if isinstance(x, jnp.ndarray):
                shape = x.shape
                if len(shape) > 0 and shape[0] % num_devices == 0:
                    walkers_per_device = shape[0] // num_devices
                    return x.reshape((num_devices, walkers_per_device) + shape[1:])
            return x
        
        # Reshape state for parallel execution
        pmapped_state = tree_map(reshape_for_pmap, state)
        
        def one_step_pmapped(
            state: SamplerState, key: jax.Array
        ) -> Tuple[SamplerState, Tuple[SamplerState, SampleStats]]:
            state, stats = self.step(key, state, tune=tune)
            return state, (state, stats)
        
        # Create parallel version
        parallel_step = jax.pmap(one_step_pmapped)
        
        # Split keys for each device
        keys = random.split(random_key, num_steps * num_devices)
        keys = keys.reshape(num_steps, num_devices, -1)
        
        if progress:
            try:
                from tqdm.auto import tqdm
                
                final_state = pmapped_state
                all_states = []
                all_stats = []
                
                with tqdm(total=num_steps, desc=progress_desc) as pbar:
                    for i in range(num_steps):
                        final_state, (trace_state, stats) = parallel_step(
                            final_state, keys[i]
                        )
                        all_states.append(trace_state)
                        all_stats.append(stats)
                        pbar.update(1)
                        
                        # Optional: Update with acceptance rate
                        if "accept_prob" in stats:
                            # Average across devices
                            accept = float(device_get(jnp.mean(stats["accept_prob"])))
                            pbar.set_postfix({"accept": f"{accept:.3f}"})
                
                # Stack and reshape results back
                trace = tree_map(lambda *xs: jnp.stack(xs), *all_states)
                sample_stats = tree_map(lambda *xs: jnp.stack(xs), *all_stats)
                
            except ImportError:
                print("tqdm not installed. Running without progress bar.")
                final_state, (trace, sample_stats) = jax.lax.scan(
                    parallel_step, pmapped_state, keys
                )
        else:
            final_state, (trace, sample_stats) = jax.lax.scan(
                parallel_step, pmapped_state, keys
            )
        
        # Reshape back to (num_steps, num_walkers, ...)
        def reshape_from_pmap(x):
            if isinstance(x, jnp.ndarray) and len(x.shape) >= 3:
                # Shape: (num_steps, num_devices, walkers_per_device, ...)
                # -> (num_steps, num_walkers, ...)
                return x.reshape((x.shape[0], -1) + x.shape[3:])
            return x
        
        final_state = tree_map(reshape_from_pmap, final_state)
        trace = tree_map(reshape_from_pmap, trace)
        sample_stats = tree_map(reshape_from_pmap, sample_stats)
        
        return Trace(
            final_state=final_state,
            samples=trace.ensemble,
            extras=trace.extras,
            move_state=trace.move_state,
            sample_stats=sample_stats,
        )