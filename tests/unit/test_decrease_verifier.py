import numpy as np
from jax import numpy as jnp

from neuralstoc.environments import v2LDSEnv
from neuralstoc.rsm.decrease_verifier import DecreaseVerifier, refine_point


def test_point_refinement():
    point = np.array([0.0, 0.0])
    scale = 0.5
    base_step = np.array([4.0, 4.0])
    coeff = 2
    new_centers, new_scales = refine_point(coeff, base_step, point, scale)
    assert len(new_centers) == 4
    assert len(new_scales) == 4
    assert np.all((new_scales - 0.25) < 1e-5)


def test_ticks2grid():
    ticks = [jnp.array([0, 1, 2]), jnp.array([0, 1])]
    grid = DecreaseVerifier.ticks2grid(ticks)
    expected_grid = jnp.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    assert jnp.array_equal(jnp.sort(grid, axis=0), jnp.sort(expected_grid, axis=0)), \
        f"Expected {expected_grid}, got {grid}"

def test_ticks2grid_3d():
    ticks = [jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([0, 1])]
    grid = DecreaseVerifier.ticks2grid(ticks)
    expected_grid = jnp.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    )
    assert jnp.array_equal(jnp.sort(grid, axis=0), jnp.sort(expected_grid, axis=0)), \
        f"Expected {expected_grid}, got {grid}"


def test_spanning_grid():
    low = jnp.array([0, 0])
    high = jnp.array([1, 2])
    size = (2, 3)
    grid = DecreaseVerifier.spanning_grid(low, high, size)
    expected_grid = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]])
    assert jnp.array_equal(jnp.sort(grid, axis=0), jnp.sort(expected_grid, axis=0)), \
        f"Expected {expected_grid}, got {grid}"


def test_spanning_grid_by_step():
    low = jnp.array([0, 0])
    high = jnp.array([1, 2])
    step = jnp.array([1, 1])
    grid = DecreaseVerifier.spanning_grid_by_step(low, high, step)
    expected_grid = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]])
    assert jnp.array_equal(jnp.sort(grid, axis=0), jnp.sort(expected_grid, axis=0)), \
        f"Expected {expected_grid}, got {grid}"


def test_grid2rectangular():
    grid = jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    size = (2, 2)
    rectangular_grid = DecreaseVerifier.grid2rectangular(grid, size)
    expected_rectangular_grid = jnp.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
    assert jnp.array_equal(rectangular_grid, expected_rectangular_grid), f"Expected {expected_rectangular_grid}, got {rectangular_grid}"


def test_grid2rectangular_3d():
    grid = jnp.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    size = (2, 2, 2)
    rectangular_grid = DecreaseVerifier.grid2rectangular(grid, size)
    expected_rectangular_grid = jnp.array([[[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]],
                                           [[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]]])
    assert jnp.array_equal(rectangular_grid, expected_rectangular_grid), f"Expected {expected_rectangular_grid}, got {rectangular_grid}"
