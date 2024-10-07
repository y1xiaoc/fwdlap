# Poor Man's Forward Laplacian (using JAX Tracer!)

This is a fan implemented (unofficial) version of the method of computing laplacian in one single forward pass, as described in the brilliant paper [Forward Laplacian: A New Computational Framework for
Neural Network-based Variational Monte Carlo](https://arxiv.org/pdf/2307.08214.pdf).

## Installation

Everything of the implementation is in this single file [`fwdlap.py`](./fwdlap.py). Feel free to copy it to the place you need. 

Alternatively, use pip to install it: 
```bash
pip install git+https://github.com/y1xiaoc/fwdlap
```
Or clone the repo and install it locally with `pip install -e .` if you need editable mode.

## Usage

The module provides two functions for forward laplacian calculation: `lap` and `lap_partial`. They are the laplacian version of `jax.jvp` and `jax.linearize` respectively. The usage of the two functions are very similar to their counterpart, but this time the tangent vector need to be batched and an extra input argument of laplacian is needed. 

For directly calculating the primal output, jacobian and laplacian, use
```python
primals_out, jacobians_out, laplacians_out = fwdlap.lap(fun, primals_in, jacobians_in, laplaicians_in)
```
Note the inputs of `fun` does not support nested pytrees. They have to be "flattened" so that primals_in is a tuple or list of arrays (or scalars).

For the partial eval version, use
```python
primals_out, lap_pe = fwdlap.lap_partial(fun, primals_in, example_jacobians_in, example_laplaicians_in)
```
Only the shapes of `example_jacobians_in` and `example_laplaicians_in` matter. After this, call `lap_pe` with the actual `jacobians_in` and `laplaicians_in` to get the actual output.
```python
jacobians_out, laplacians_out = lap_pe(jacobians_in, laplaicians_in)
```

Please check the [docstrings](./fwdlap.py#L42) of these two functions for more details. The [test](./test_fwdlap.py#L65) file also contains some examples of usage, including passing symbolic zeros.

## Why this implementation?

The method proposed in the Forward Laplacian paper is in fact very similar to the existing (yet experimental) module [`jet`](https://jax.readthedocs.io/en/latest/jax.experimental.jet.html) in jax, up to the second order. The propagation rules are almost identical, with only one difference that in forward laplacian, the jacobian contribution to the laplacian (first term in eq. 17 of the paper) is summed over for every operation, while in jet it is simply `vmap`'ed and summed at end of the pass. (See [this discussion](https://github.com/google/jax/discussions/9598) for how to use jet to calculate laplacian.) This difference makes forward laplacian consume less memory comparing to `vmap`'ed jet, and may save some computation time as well (at a cost of doing a reduced sum for every operation).

Given the similarity of the two methods, I tried to implement the forward laplacian method using jax tracer, taking reference on the `jet` module. However, the implementation of `jet` is very inefficient, because it will always instantiate all the symbolic zeros. Therefore, I wrote my own tracer without using any jet rules, but simply `jvp` twice for 2nd order derivatives and make all `Zero`s pass through. The result is this module, [`fwdlap.py`](./fwdlap.py)!

Comparing to the proposed implementation in the paper, which overloads all `jax.numpy` operators, this implementation works on the jax primitive level, reusing all the jvp rules and let jax compiler do the trick. This approach is much cheaper in terms of coding cost, and that's why I call it _"poor man's"_ version. In addition, it is also more flexible, as it can in principle handle any jax function, not limited to the operators overloaded in `jax.numpy`. The drawback is I did not spend much time on optimizing the forward rule for each operator. However, thanks to the powerful compiler of jax (and my careful treatment of symbolic zeros), most of these optimization are not necessary (such as those for linear or element-wise operators). The bilinear operators are the only exceptions, for which I implemented a special rule in the tracer to take advantage of the symmetry of the Hessian matrix. 

At the time of writing, the performance comparison with the official version is not available, as the official one has not been released yet and I have no access to it.

## Example on kinetic energy

Here's an example of using the `fwdlap` module to calculate the kinetic energy of a given log of wavefunction `log_psi`. It supports (mini_batched) loop evaluation in both the batch dimension (`batch_size`) and the inner jacobian dimension (`inner_size`). Set them to `None` will use the full batch version. Choosing these two parameters carefully, this implementation can achieve 3x speed up on some attention based neural network wavefunctions, comparing to the old one used in the ferminet repo. It also saves memory as there's no need to store the intermediate results of backward propagation.

```python
import jax
from jax import lax
from jax import numpy as jnp

import fwdlap

def calc_ke_fwdlap(log_psi, x, inner_size=None, batch_size=None):
    # calc -0.5 * (\nable^2 \psi) / \psi
    # handle batch of x automatically
    def _lapl_over_psi(x):
        # (\nable^2 f) / f = \nabla^2 log|f| + (\nabla log|f|)^2
        # x is assumed to have shape [n_ele, n_dim], not batched
        x_shape = x.shape
        flat_x = x.reshape(-1)
        ncoord = flat_x.size
        f = lambda flat_x: log_psi(flat_x.reshape(x_shape)) # take flattened x
        eye = jnp.eye(ncoord, dtype=x.dtype)
        zero = fwdlap.zero_tangent_from_primal(flat_x)
        if inner_size is None:
            primals, grads, laps = fwdlap.lap(f, (flat_x,), (eye,), (zero,))
            laplacian = (grads**2).sum() + laps
        else:
            eye = eye.reshape(ncoord//inner_size, inner_size, ncoord)
            primals, f_lap_pe = fwdlap.lap_partial(f, (flat_x,), (eye[0],), (zero,))
            def loop_fn(i, val):
                (jac, lap) = f_lap_pe((eye[i],), (zero,))
                val += (jac**2).sum() + lap
                return val
            laplacian = lax.fori_loop(0, ncoord//inner_size, loop_fn, 0.0)
        return laplacian
    # handle batch of x, assuming x has at most 3 dims
    if x.ndim <= 2:
        return -0.5 * _lapl_over_psi(x)
    if x.ndim != 3:
        msg = f"only support x with ndim less than 3, get {x.ndim}"
        raise ValueError(msg)
    # batched version when x.ndim == 3
    lapl_fn = jax.vmap(_lapl_over_psi)
    if batch_size is None:
        return -0.5 * lapl_fn(x)
    x_batched = x.reshape(x.shape[0]//batch_size, batch_size, *x.shape[1:])
    return -0.5 * lax.map(lapl_fn, x_batched).reshape(-1)
```



