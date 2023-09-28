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
from jax.dtypes import float0
try:
    from jax.extend import linear_util as lu
except ImportError:
    from jax import linear_util as lu
from jax.util import split_list, safe_map as smap
from jax.interpreters.ad import Zero, instantiate_zeros

from jax._src.util import unzip3


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

    @lu.transformation_with_aux
    def flatten_fun_output(*args):
        ans = yield args, {}
        yield tree_flatten(ans)

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
        jsize = self.main.jsize
        zero_jac = Zero(core.get_aval(
                jnp.expand_dims(val, 0).repeat(jsize, axis=0)
            ).at_least_vspace())
        zero_lap = Zero(core.get_aval(val).at_least_vspace())
        return LapTracer(self, val, zero_jac, zero_lap)

    def lift(self, val):
        jsize = self.main.jsize
        zero_jac = Zero(core.get_aval(
                jnp.expand_dims(val, 0).repeat(jsize, axis=0)
            ).at_least_vspace())
        zero_lap = Zero(core.get_aval(val).at_least_vspace())
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
                outs = jvp(*p_in, *t_in)
                return split_list(outs, [len(outs) // 2])
            return wrapped
        primals_out, jacs_out, laps_out = hvv_by_jvp(
            wrap_jvp(jvp.call_wrapped), primals_in, jacs_in, laps_in,
            inner_jvp=wrap_jvp(jvp.f))
        return [LapTracer(self, p, j, l)
                for p, j, l in zip(primals_out, jacs_out, laps_out)]

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        del primitive, fwd, bwd, out_trees  # Unused.
        return fun.call_wrapped(*tracers)


call_param_updaters: dict[core.Primitive, Callable[..., Any]] = {}


def hvv_by_jvp(f_jvp, primals_in, jacs_in, laps_in, inner_jvp=None):
    z0 = primals_in
    z1 = jax.tree_map(recast_np_float0, primals_in, jacs_in)
    z2 = jax.tree_map(recast_np_float0, primals_in, laps_in)
    if inner_jvp is None:
        inner_jvp = f_jvp
    def hvv(v):
        inner = lambda *a: inner_jvp(a, v)
        (_, o1), (_, o2_1) = jax.jvp(inner, z0, v)
        return o1, o2_1
    o1, o2_1 = jax.vmap(hvv, in_axes=0, out_axes=0)(z1)
    o0, o2_2 = f_jvp(z0, z2)
    add_o2 = lambda a, b: (b if type(a) is Zero else a.sum(0)
                           if type(b) is Zero else a.sum(0) + b)
    o2 = jax.tree_map(add_o2, o2_1, o2_2)
    return o0, o1, o2


def primitive_by_jvp(primitive, primals_in, jacs_in, laps_in, **params):
    func = partial(primitive.bind, **params)
    f_jvp = partial(jax.jvp, func)
    return hvv_by_jvp(f_jvp, primals_in, jacs_in, laps_in, inner_jvp=None)


def recast_np_float0(primal, tangent):
    tangent = instantiate_zeros(tangent)
    if core.primal_dtype_to_tangent_dtype(jnp.result_type(primal)) == float0:
        return np.zeros(tangent.shape, dtype=float0)
    else:
        return tangent

### rule definitions

lap_rules = {}

