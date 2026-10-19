# mypy: ignore-errors

from itertools import product

import jax.numpy as jnp
import pytest
from jax import random, vmap
from jax.flatten_util import ravel_pytree
from scipy import stats

from emcee_jax import EnsembleSampler, moves

# Each `assert pvalue > KS_ALPHA` below is a hypothesis test against a fixed
# seed, so a correct sampler is still expected to trip a fixed-alpha cutoff
# occasionally just from sampling variability. This module runs 24 such KS
# tests across test_uniform's and test_normal's parametrizations (12 each),
# so a naive per-test alpha of 0.01 gives a ~1 - 0.99**24 ≈ 21% chance that
# some case fails on any given CI run even when the sampler is unbiased.
# Bonferroni-correct across all 24 tests to keep the overall false-failure
# rate at ~1%.
_N_KS_TESTS = 24
KS_ALPHA = 0.01 / _N_KS_TESTS


@pytest.mark.parametrize(
    "ndim,move",
    product(
        [1, 2],
        [
            moves.Stretch(),
            moves.DiffEvol(),
            moves.DiffEvolSlice(),
            moves.compose(moves.Stretch(), moves.DiffEvolSlice()),
        ],
    ),
)
def test_uniform(ndim, move, seed=1, num_walkers=32, num_steps=2_000):
    key = random.PRNGKey(seed)
    coords_key, init_key, sample_key = random.split(key, 3)
    coords = random.uniform(coords_key, shape=(num_walkers, ndim))
    sampler = EnsembleSampler(
        lambda x: jnp.sum(jnp.where((0 < x) & (x < 1), 0.0, -jnp.inf)),
        move=move,
    )
    state = sampler.init(init_key, coords)
    trace = sampler.sample(sample_key, state, num_steps)
    flat_samples = vmap(vmap(lambda x: ravel_pytree(x)[0]))(
        trace.samples.coordinates
    )
    assert flat_samples.shape == (num_steps, num_walkers, ndim)
    for n in range(ndim):
        _, pvalue = stats.kstest(
            flat_samples[::100, :, n].flatten(), "uniform"
        )
        assert pvalue > KS_ALPHA, n


@pytest.mark.parametrize(
    "pytree,move",
    product(
        [True, False],
        [moves.Stretch(), moves.DiffEvol(), moves.DiffEvolSlice()],
    ),
)
def test_normal(pytree, move, seed=1, num_walkers=32, num_steps=2_000):
    def log_prob(x):
        if pytree:
            x, y = x["x"], x["y"]
        else:
            x, y = x
        return -0.5 * (x**2 + y**2)

    key = random.PRNGKey(seed)
    coords_key, init_key, sample_key = random.split(key, 3)
    coords = random.normal(coords_key, shape=(num_walkers, 2))
    if pytree:
        coords = {"x": coords[:, 0], "y": coords[:, 1]}
    sampler = EnsembleSampler(log_prob, move=move)
    state = sampler.init(init_key, coords)
    trace = sampler.sample(sample_key, state, num_steps)
    flat_samples = vmap(vmap(lambda x: ravel_pytree(x)[0]))(
        trace.samples.coordinates
    )
    assert flat_samples.shape == (num_steps, num_walkers, 2)
    for n in range(2):
        _, pvalue = stats.kstest(flat_samples[::100, :, n].flatten(), "norm")
        assert pvalue > KS_ALPHA, n
