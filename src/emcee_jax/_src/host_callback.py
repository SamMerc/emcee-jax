from functools import wraps
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes
from jax.experimental import pure_callback
from jax.tree_util import tree_flatten

from emcee_jax._src.log_prob_fn import LogProbFn
from emcee_jax._src.types import Array, PyTree


def wrap_python_log_prob_fn(
    python_log_prob_fn: Callable[..., Array]
) -> LogProbFn:
    """Wrap a pure Python log probability function for use with JAX.
        
        This uses jax.pure_callback (modern JAX) to call pure Python functions
        from within JAX transformations.
        
        Args:
            python_log_prob_fn: A Python function that computes log probability.
                Can use numpy instead of jax.numpy.
        
        Returns:
            A JAX-compatible log probability function.
        """
    
    @wraps(python_log_prob_fn)
    def log_prob_fn(params: Array) -> Array:
        """Evaluate log prob for a single set of parameters."""
        # Infer output dtype from input
        dtype = _tree_dtype(params)
        # Use pure_callback to call the Python function
        return pure_callback(
            python_log_prob_fn,
            jax.ShapeDtypeStruct((), dtype),  # Output shape and dtype
            params,
            vectorized=False
        )
    # Return vectorized version that works with ensembles
    return jax.vmap(log_prob_fn)


def _tree_dtype(tree: PyTree) -> Any:
    """Infer dtype from a PyTree."""
    leaves, _ = tree_flatten(tree)
    from_dtypes = [dtypes.dtype(l) for l in leaves]
    return dtypes.result_type(*from_dtypes)


def _arraylike(x: Array) -> bool:
    """Check if x is array-like."""
    return (
        isinstance(x, np.ndarray)
        or isinstance(x, jnp.ndarray)
        or hasattr(x, "__jax_array__")
        or np.isscalar(x)
    )