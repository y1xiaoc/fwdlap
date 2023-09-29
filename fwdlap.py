# Copyright 2020 The JAX Authors.
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

from typing import Any, Callable

from functools import partial

import numpy as np

import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import (tree_structure, treedef_is_leaf,
                           tree_flatten, tree_unflatten,)

from jax import core
try:
    from jax.extend import linear_util as lu
except ImportError:
    from jax import linear_util as lu
from jax.util import split_list, safe_map as smap
from jax.interpreters import ad
from jax.interpreters.ad import Zero


def lap(fun, primals, jacobians, laplacians):
    try:
        jsize, = set(map(lambda x:x.shape[0], jacobians))
    except ValueError:
        msg = "jacobians have inconsistent first dimensions for different arguments"
        raise ValueError(msg) from None

    for i, (x, j, l) in enumerate(zip(primals, jacobians, laplacians)):
        for t, name in ((x, "primal"), (j, "jacobian"), (l, "laplacian")):
            treedef = tree_structure(t)
            if not treedef_is_leaf(treedef):
                raise ValueError(f"{name} value at position {i} is not an array")

    f, out_tree = flatten_fun_output(lu.wrap_init(fun))
    out_primals, out_jacs, out_laps = lap_fun(lap_subtrace(f), jsize).call_wrapped(
        primals, jacobians, laplacians)
    return (tree_unflatten(out_tree(), out_primals),
            tree_unflatten(out_tree(), out_jacs),
            tree_unflatten(out_tree(), out_laps))


@lu.transformation
def lap_fun(jsize, primals, jacobians, laplacians):
    with core.new_main(LapTrace) as main:
        main.jsize = jsize
        out_primals, out_jacs, out_laps = yield (main, primals, jacobians, laplacians), {}
        del main
    out_jacs = [jnp.zeros((jsize, *p.shape), p.dtype) if type(j) is Zero else j
                for p, j in zip(out_primals, out_jacs)]
    out_laps = [jnp.zeros_like(p) if type(l) is Zero else l
                for p, l in zip(out_primals, out_laps)]
    yield out_primals, out_jacs, out_laps


@lu.transformation
def lap_subtrace(main, primals, jacobians, laplacians):
    trace = LapTrace(main, core.cur_sublevel())
    in_tracers = map(partial(LapTracer, trace), primals, jacobians, laplacians)
    ans = yield in_tracers, {}
    out_tracers = map(trace.full_raise, ans)
    out_primals, out_jacs, out_laps = unzip3((t.primal, t.jacobian, t.laplacian)
                                             for t in out_tracers)
    yield out_primals, out_jacs, out_laps


@lu.transformation_with_aux
def traceable(in_tree_def, *primals_jacs_laps):
    primals_in, jacs_in, laps_in = tree_unflatten(in_tree_def, primals_jacs_laps)
    primals_out, jacs_out, laps_out = yield (primals_in, jacs_in, laps_in), {}
    out_flat, out_tree_def = tree_flatten((primals_out, jacs_out, laps_out))
    yield out_flat, out_tree_def


class LapTracer(core.Tracer):
    __slots__ = ["primal", "jacobian", "laplacian"]

    def __init__(self, trace, primal, jacobian, laplacian):
        self._trace = trace
        self.primal = primal
        self.jacobian = jacobian
        self.laplacian = laplacian

    @property
    def aval(self):
        return core.get_aval(self.primal)

    def full_lower(self):
        if type(self.jacobian) is Zero and type(self.laplacian) is Zero:
            return core.full_lower(self.primal)
        else:
            return self


class LapTrace(core.Trace):

    def pure(self, val):
        zero_jac = zero_tangent_from_primal(val)
        zero_lap = zero_tangent_from_primal(val)
        return LapTracer(self, val, zero_jac, zero_lap)

    def lift(self, val):
        zero_jac = zero_tangent_from_primal(val)
        zero_lap = zero_tangent_from_primal(val)
        return LapTracer(self, val, zero_jac, zero_lap)

    def sublift(self, val):
        return LapTracer(self, val.primal, val.jacobian, val.laplacian)

    def process_primitive(self, primitive, tracers, params):
        primals_in, jacs_in, laps_in = unzip3((t.primal, t.jacobian, t.laplacian)
                                              for t in tracers)
        if primitive in lap_rules:
            rule = lap_rules[primitive]
            primal_out, jac_out, lap_out = rule(
                primals_in, jacs_in, laps_in, **params)
        else:
            primal_out, jac_out, lap_out = primitive_by_jvp(
                primitive, primals_in, jacs_in, laps_in, **params)
        if not primitive.multiple_results:
            return LapTracer(self, primal_out, jac_out, lap_out)
        else:
            return [LapTracer(self, p, j, l)
                    for p, j, l in zip(primal_out, jac_out, lap_out)]

    def process_call(self, call_primitive, f, tracers, params):
        primals_in, jacs_in, laps_in = unzip3((t.primal, t.jacobian, t.laplacian)
                                              for t in tracers)
        primals_jacs_laps, in_tree_def = tree_flatten((primals_in, jacs_in, laps_in))
        f_jet, out_tree_def = traceable(lap_subtrace(f, self.main), in_tree_def)
        update_params = call_param_updaters.get(call_primitive)
        new_params = (update_params(params, len(primals_jacs_laps))
                      if update_params else params)
        result = call_primitive.bind(f_jet, *primals_jacs_laps, **new_params)
        primals_out, jacs_out, laps_out = tree_unflatten(out_tree_def(), result)
        return [LapTracer(self, p, j, l)
                for p, j, l in zip(primals_out, jacs_out, laps_out)]

    def post_process_call(self, call_primitive, out_tracers, params):
        primals, jacs, laps = unzip3((t.primal, t.jacobian, t.laplacian)
                                     for t in out_tracers)
        out, treedef = tree_flatten((primals, jacs, laps))
        del primals, jacs, laps
        main = self.main
        def todo(x):
            primals, jacs, laps = tree_unflatten(treedef, x)
            trace = LapTrace(main, core.cur_sublevel())
            return map(partial(LapTracer, trace), primals, jacs, laps)
        return out, todo

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *,
                                symbolic_zeros):
        primals_in, jacs_in, laps_in = unzip3((t.primal, t.jacobian, t.laplacian)
                                              for t in tracers)
        primals_in = smap(core.full_lower, primals_in)
        def wrap_jvp(jvp):
            def wrapped(p_in, t_in):
                t_in = smap(ad.instantiate_zeros, t_in)
                t_in = smap(ad.replace_float0s, p_in, t_in)
                outs = jvp(*p_in, *t_in)
                p_out, t_out = split_list(outs, [len(outs) // 2])
                t_out = smap(ad.recast_to_float0, p_out, t_out)
                return p_out, t_out
            return wrapped
        primals_out, jacs_out, laps_out = vhv_by_jvp(
            wrap_jvp(jvp.call_wrapped), primals_in, jacs_in, laps_in,
            inner_jvp=wrap_jvp(jvp.f))
        return [LapTracer(self, p, j, l)
                for p, j, l in zip(primals_out, jacs_out, laps_out)]

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        del primitive, fwd, bwd, out_trees  # Unused.
        return fun.call_wrapped(*tracers)


call_param_updaters: dict[core.Primitive, Callable[..., Any]] = {}


def my_jvp(fun, primals, tangents):
    # this jvp is transparant to Zero, and assumes flattened input
    f, out_tree = flatten_fun_output(lu.wrap_init(fun))
    jvp_f = ad.jvp(f, instantiate=False)
    out_primals, out_tangents = jvp_f.call_wrapped(primals, tangents)
    out_tree = out_tree()
    return (tree_unflatten(out_tree, out_primals),
            tree_unflatten(out_tree, out_tangents))


def vhv_by_jvp(f_jvp, primals_in, jacs_in, laps_in, inner_jvp=None):
    z0, z1, z2 = primals_in, jacs_in, laps_in
    if inner_jvp is None:
        inner_jvp = f_jvp
    def hvv(v):
        inner = lambda *a: inner_jvp(a, v)
        (_, o1), (_, o2_1) = my_jvp(inner, z0, v)
        return o1, o2_1
    # second term in laplacian
    o0, o2_2 = f_jvp(z0, z2)
    multi_out = not treedef_is_leaf(tree_structure(o0))
    # jacobian and first term in laplacian, handle all empty case
    if all(type(j) is Zero for j in z1):
        o1 = [zero_tangent_from_primal(p) for p in o0]
        o2 = o2_2
    else:
        o1, o2_1 = jax.vmap(hvv, in_axes=0, out_axes=0)(z1)
        add_o2 = lambda a, b: (b if type(a) is Zero else a.sum(0)
                            if type(b) is Zero else a.sum(0) + b)
        o2 = smap(add_o2, o2_1, o2_2) if multi_out else add_o2(o2_1, o2_2)
    return o0, o1, o2


def primitive_by_jvp(primitive, primals_in, jacs_in, laps_in, **params):
    func = partial(primitive.bind, **params)
    f_jvp = partial(my_jvp, func)
    return vhv_by_jvp(f_jvp, primals_in, jacs_in, laps_in, inner_jvp=None)


def zero_tangent_from_primal(primal):
    return Zero(core.get_aval(primal).at_least_vspace())


@lu.transformation_with_aux
def flatten_fun_output(*args):
    ans = yield args, {}
    yield tree_flatten(ans)


def unzip3(xyzs) :
  """Unzip sequence of length-3 tuples into three tuples."""
  # copied from jax._src.util, remove type annotations
  xs, ys, zs = [], [], []
  for x, y, z in xyzs:
    xs.append(x)
    ys.append(y)
    zs.append(z)
  return tuple(xs), tuple(ys), tuple(zs)


### rule definitions

lap_rules = {}


def defzero(prim):
    lap_rules[prim] = partial(zero_prop, prim)

def zero_prop(prim, primals_in, jacs_in, laps_in, **params):
    primal_out = prim.bind(*primals_in, **params)
    jac_out = zero_tangent_from_primal(primal_out)
    lap_out = zero_tangent_from_primal(primal_out)
    return primal_out, jac_out, lap_out

defzero(lax.le_p)
defzero(lax.lt_p)
defzero(lax.gt_p)
defzero(lax.ge_p)
defzero(lax.eq_p)
defzero(lax.ne_p)
defzero(lax.not_p)
defzero(lax.and_p)
defzero(lax.or_p)
defzero(lax.xor_p)
defzero(lax.floor_p)
defzero(lax.ceil_p)
defzero(lax.round_p)
defzero(lax.sign_p)
defzero(lax.stop_gradient_p)
defzero(lax.is_finite_p)
defzero(lax.shift_left_p)
defzero(lax.shift_right_arithmetic_p)
defzero(lax.shift_right_logical_p)
defzero(lax.bitcast_convert_type_p)


def deflinear(prim):
    lap_rules[prim] = partial(linear_prop, prim)

def linear_prop(prim, primals_in, jacs_in, laps_in, **params):
    pprim = partial(prim.bind, **params)
    primal_out, lap_out = my_jvp(pprim, primals_in, laps_in)
    if all(type(j) is Zero for j in jacs_in):
        jac_out = zero_tangent_from_primal(primal_out)
    else:
        wrapped = lambda t: pprim(*smap(ad.instantiate_zeros, t))
        jac_out = jax.vmap(wrapped, 0, 0)(jacs_in)
    return primal_out, jac_out, lap_out

deflinear(lax.neg_p)
deflinear(lax.real_p)
deflinear(lax.complex_p)
deflinear(lax.conj_p)
deflinear(lax.imag_p)
deflinear(lax.add_p)
deflinear(ad.add_jaxvals_p)
deflinear(lax.sub_p)
deflinear(lax.convert_element_type_p)
deflinear(lax.broadcast_in_dim_p)
deflinear(lax.concatenate_p)
deflinear(lax.pad_p)
deflinear(lax.reshape_p)
deflinear(lax.squeeze_p)
deflinear(lax.rev_p)
deflinear(lax.transpose_p)
deflinear(lax.slice_p)
deflinear(lax.reduce_sum_p)
deflinear(lax.reduce_window_sum_p)
deflinear(lax.fft_p)
deflinear(lax.device_put_p)


def defelemwise(prim):
    lap_rules[prim] = partial(elemwise_prop, prim)

def elemwise_prop(prim, primals_in, jacs_in, laps_in, **params):
    print(prim)
    pprim = partial(prim.bind, **params)
    z0, z1, z2 = primals_in, jacs_in, laps_in
    o0, o2_2 = my_jvp(pprim, z0, z2)
    if all(type(j) is Zero for j in jacs_in):
        o1 = zero_tangent_from_primal(o0)
        return o0, o1, o2_2
    # now jacs are not all zero
    o1 = jax.vmap(lambda z: my_jvp(pprim, z0, z)[1], 0, 0)(z1)
    nonzero_idx = [i for i, jac in enumerate(z1) if type(jac) is not Zero]
    hess_fn = jax.hessian(pprim, argnums=nonzero_idx)
    for _ in range(o0.ndim):
        hess_fn = jax.vmap(hess_fn)
    hess_mat = hess_fn(*z0)
    o2 = o2_2
    for i, nzi in enumerate(nonzero_idx):
        for j, nzj in enumerate(nonzero_idx[i:]):
            hess = hess_mat[i][j]
            if i != j:
                hess *= 2
            o2 += (hess * z1[nzi] * z1[nzj]).sum(0)
    return o0, o1, o2

defelemwise(lax.sin_p)
defelemwise(lax.cos_p)
