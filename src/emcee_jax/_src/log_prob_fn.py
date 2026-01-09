from typing import Any, Callable, Tuple, Union
import jax
import jax.numpy as jnp
from emcee_jax._src.types import Array, PyTree

LogProbFn = Callable[..., Union[Array, Tuple[Array, PyTree]]]


class WrappedLogProbFn:
    """Wrapper for log probability functions that handles deterministics and NaNs."""
    
    def __init__(
        self, 
        log_prob_fn: LogProbFn, 
        *log_prob_args: Any, 
        **log_prob_kwargs: Any
    ):
        self.log_prob_fn = log_prob_fn
        self.log_prob_args = log_prob_args
        self.log_prob_kwargs = log_prob_kwargs
    
    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[Array, Any]:
        """Call the wrapped function and handle deterministics and NaNs."""
        # Merge the stored args/kwargs with the call-time args/kwargs
        all_args = args + self.log_prob_args
        all_kwargs = {**self.log_prob_kwargs, **kwargs}
        
        # Call the original function
        result = self.log_prob_fn(*all_args, **all_kwargs)
        
        # Unwrap deterministics if they are provided or default to None
        if isinstance(result, tuple):
            log_prob, *deterministics = result
            if len(deterministics) == 1:
                deterministics = deterministics[0]
        else:
            log_prob = result
            deterministics = None
        
        if log_prob is None:
            raise ValueError(
                "A log probability function must return a scalar value, got None"
            )
        
        try:
            log_prob = jnp.reshape(log_prob, ())
        except TypeError:
            raise ValueError(
                "A log probability function must return a scalar; "
                f"computed shape is '{log_prob.shape}', expected '()'"
            )
        
        # Handle the case where the computed log probability is NaN by replacing it
        # with negative infinity so that it gets rejected
        log_prob = jax.lax.cond(
            jnp.isnan(log_prob), lambda: -jnp.inf, lambda: log_prob
        )
        
        return log_prob, deterministics


def wrap_log_prob_fn(
    log_prob_fn: LogProbFn, *log_prob_args: Any, **log_prob_kwargs: Any
) -> WrappedLogProbFn:
    """Wrap a log probability function to handle deterministics and NaNs.
    
    Args:
        log_prob_fn: The log probability function to wrap
        *log_prob_args: Positional arguments to pass to the function
        **log_prob_kwargs: Keyword arguments to pass to the function
    
    Returns:
        A wrapped function that handles deterministics and NaN values
    """
    return WrappedLogProbFn(log_prob_fn, *log_prob_args, **log_prob_kwargs)