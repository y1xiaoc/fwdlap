# Copyright 2023 Yixiao Chen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The stax network is adapted from readme in jax repo

from functools import partial

import pytest
import numpy as np

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.example_libraries.stax import (
    Conv, Dense, MaxPool, Relu, Tanh, Flatten, Softplus)

import fwdlap


def get_network():
    # Use stax to set up network initialization and evaluation functions
    net_init, net_apply = stax.serial(
        Conv(32, (3, 3), padding='SAME'), Relu,
        Conv(64, (3, 3), padding='SAME'), Relu,
        MaxPool((2, 2)), Flatten,
        Dense(128), Tanh,
        Dense(10), Softplus,
    )
    # Initialize parameters, no batch shape
    rng = jax.random.PRNGKey(0)
    in_shape = (1, 5, 5, 2)
    out_shape, net_params = net_init(rng, in_shape)
    return net_apply, net_params, in_shape, out_shape


@pytest.fixture(scope="module")
def common_data():
    key = jax.random.PRNGKey(1)
    net_apply, net_params, in_shape, out_shape = get_network()
    net_fn = jax.jit(partial(net_apply, net_params)) # mysterious error if no jit
    x = jax.random.normal(key, in_shape)
    jac = jax.jacfwd(net_fn)(x)
    hess = jax.hessian(net_fn)(x)
    lap = hess.reshape(*out_shape, x.size, x.size).trace(0, -1, -2)
    return net_fn, x, jac, lap


@pytest.mark.parametrize("symbolic_zero", [True, False])
@pytest.mark.parametrize("use_jit", [True, False])
def test_lap(common_data, symbolic_zero, use_jit):
    net_fn, x, jac_target, lap_target = common_data
    eye = jnp.eye(x.size).reshape(x.size, *x.shape)
    zero = (fwdlap.zero_tangent_from_primal(x)
            if symbolic_zero else jnp.zeros_like(x))
    flap_fn = (jax.jit(partial(fwdlap.lap, net_fn))
               if use_jit else partial(fwdlap.lap, net_fn))
    out, jac, lap = flap_fn((x,), (eye,), (zero,))
    jac = jnp.moveaxis(jac, -1, 0).reshape(*out.shape, *x.shape)
    np.testing.assert_allclose(out, net_fn(x), atol=1e-5)
    np.testing.assert_allclose(jac, jac_target, atol=1e-5)
    np.testing.assert_allclose(lap, lap_target, atol=1e-5)


@pytest.mark.parametrize("symbolic_zero", [True, False])
@pytest.mark.parametrize("use_jit", [True, False])
def test_lap_partial(common_data, symbolic_zero, use_jit):
    net_fn, x, jac_target, lap_target = common_data
    eye = jnp.eye(x.size).reshape(x.size, *x.shape)
    zero = (fwdlap.zero_tangent_from_primal(x)
            if symbolic_zero else jnp.zeros_like(x))
    out, lap_pe = fwdlap.lap_partial(net_fn, (x,), (eye,), (zero,))
    lap_fn = jax.jit(lap_pe) if use_jit else lap_pe
    jac, lap = lap_fn((eye,), (zero,))
    jac = jnp.moveaxis(jac, -1, 0).reshape(*out.shape, *x.shape)
    np.testing.assert_allclose(out, net_fn(x), atol=1e-5)
    np.testing.assert_allclose(jac, jac_target, atol=1e-5)
    np.testing.assert_allclose(lap, lap_target, atol=1e-5)


@pytest.mark.parametrize("symbolic_zero", [True, False])
@pytest.mark.parametrize("use_jit", [True, False])
def test_lap_vmapped(common_data, symbolic_zero, use_jit):
    net_fn, x, _, lap_target = common_data
    vnet_fn = jax.vmap(net_fn, in_axes=0, out_axes=0)
    x = jnp.stack([x, x])
    lap_target = jnp.stack([lap_target, lap_target])
    eye = jnp.eye(x.size).reshape(x.size, *x.shape)
    zero = (fwdlap.zero_tangent_from_primal(x)
            if symbolic_zero else jnp.zeros_like(x))
    flap_fn = (jax.jit(partial(fwdlap.lap, vnet_fn))
               if use_jit else partial(fwdlap.lap, vnet_fn))
    _, _, lap = flap_fn((x,), (eye,), (zero,))
    np.testing.assert_allclose(lap, lap_target, atol=1e-5)
