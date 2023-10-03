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

from typing import Any, Callable, Sequence, Union

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
from jax.interpreters import partial_eval as pe
from jax.interpreters.ad import Zero

from jax._src.util import unzip3, weakref_lru_cache
from jax.experimental.pjit import pjit_p


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
    out_primals, out_jacs, out_laps = lap_fun(
        lap_subtrace(f), jsize, True).call_wrapped(
        primals, jacobians, laplacians)
    return (tree_unflatten(out_tree(), out_primals),
            tree_unflatten(out_tree(), out_jacs),
            tree_unflatten(out_tree(), out_laps))


@lu.transformation
def lap_fun(jsize, instantiate, primals, jacobians, laplacians):
    with core.new_main(LapTrace) as main:
        main.jsize = jsize
        out_primals, out_jacs, out_laps = yield (main, primals, jacobians, laplacians), {}
        del main
    if type(instantiate) is bool: # noqa: E721
        instantiate = [instantiate] * len(out_jacs)
    out_jacs = [jnp.zeros((jsize, *p.shape), p.dtype)
                if type(j) is Zero and inst else j
                for p, j, inst in zip(out_primals, out_jacs, instantiate)]
    out_laps = [jnp.zeros_like(p)
                if type(l) is Zero and inst else l
                for p, l, inst in zip(out_primals, out_laps, instantiate)]
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
        primals_out, jacs_out, laps_out = vhv_by_jvp(
            wrap_custom_jvp(jvp).call_wrapped, primals_in, jacs_in, laps_in,
            inner_jvp=wrap_custom_jvp(lu.wrap_init(jvp.f)).call_wrapped)
        return [LapTracer(self, p, j, l)
                for p, j, l in zip(primals_out, jacs_out, laps_out)]

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        del primitive, fwd, bwd, out_trees  # Unused.
        return fun.call_wrapped(*tracers)


call_param_updaters: dict[core.Primitive, Callable[..., Any]] = {}


def zero_tangent_from_primal(primal):
    return Zero(core.get_aval(primal).at_least_vspace())


@lu.transformation_with_aux
def flatten_fun_output(*args):
    ans = yield args, {}
    yield tree_flatten(ans)


@lu.transformation
def wrap_custom_jvp(primals_in, tangents_in):
    tangents_in = smap(ad.instantiate_zeros, tangents_in)
    tangents_in = smap(ad.replace_float0s, primals_in, tangents_in)
    ans = yield (*primals_in, *tangents_in), {}
    primals_out, tangents_out = split_list(ans, [len(ans) // 2])
    tangents_out = smap(ad.recast_to_float0, primals_out, tangents_out)
    yield primals_out, tangents_out


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
    def vhv(v):
        inner = lambda *a: inner_jvp(a, v)[1]
        return my_jvp(inner, z0, v)
    # second term in laplacian
    o0, o2_2 = f_jvp(z0, z2)
    multi_out = not treedef_is_leaf(tree_structure(o0))
    # jacobian and first term in laplacian, handle all empty case
    if all(type(j) is Zero for j in z1):
        o1 = ([zero_tangent_from_primal(p) for p in o0]
              if multi_out else zero_tangent_from_primal(o0))
        o2 = o2_2
    else:
        o1, o2_1 = jax.vmap(vhv, in_axes=0, out_axes=0)(z1)
        add_o2 = lambda a, b: (b if type(a) is Zero else a.sum(0)
                               if type(b) is Zero else a.sum(0) + b)
        o2 = smap(add_o2, o2_1, o2_2) if multi_out else add_o2(o2_1, o2_2)
    return o0, o1, o2


def primitive_by_jvp(primitive, primals_in, jacs_in, laps_in, **params):
    func = partial(primitive.bind, **params)
    f_jvp = partial(my_jvp, func)
    return vhv_by_jvp(f_jvp, primals_in, jacs_in, laps_in, inner_jvp=None)


### rule definitions

lap_rules = {}


def defmultivar(prim):
    lap_rules[prim] = partial(multivar_prop, prim)

def multivar_prop(prim, primals_in, jacs_in, laps_in, **params):
    # print("multivar rule", prim)
    pprim = partial(prim.bind, **params)
    z0, z1, z2 = primals_in, jacs_in, laps_in
    o0, o2_2 = my_jvp(pprim, z0, z2)
    if all(type(j) is Zero for j in jacs_in):
        o1 = zero_tangent_from_primal(o0)
        return o0, o1, o2_2
    o1 = jax.vmap(lambda v: my_jvp(pprim, z0, v), 0, 0)(z1)[1]
    mul2 = lambda x: 2*x if type(x) is not Zero else x
    add_o2 = lambda a, b: (b if type(a) is Zero else a.sum(0)
                           if type(b) is Zero else a.sum(0) + b)
    def vhv(v1, v2):
        inner = lambda *a: my_jvp(pprim, a, v1)[1]
        return my_jvp(inner, z0, v2)[1]
    o2 = o2_2
    for i in range(len(primals_in)):
        fold_z1 = [zero_tangent_from_primal(p)
                   if j < i else mul2(t) if j > i else t
                   for j, (p, t) in enumerate(zip(z0,z1))]
        diag_z1 = [zero_tangent_from_primal(p) if j != i else t
                   for j, (p, t) in enumerate(zip(z0,z1))]
        o2_1_slice = (jax.vmap(vhv, in_axes=0, out_axes=0)(fold_z1, diag_z1)
                      if tree_flatten((fold_z1, diag_z1))[0] else
                      zero_tangent_from_primal(o0))
        o2 = add_o2(o2_1_slice, o2)
    return o0, o1, o2

defmultivar(lax.mul_p)
defmultivar(lax.div_p)
defmultivar(lax.dot_general_p)


def lap_jaxpr(jaxpr: core.ClosedJaxpr,
              jsize: int,
              nonzeros1: Sequence[bool],
              nonzeros2: Sequence[bool],
              instantiate: Union[bool, Sequence[bool]]
              ) -> tuple[core.ClosedJaxpr, list[bool]]:
    if type(instantiate) is bool: # noqa: E721
        instantiate = (instantiate,) * len(jaxpr.out_avals)
    return _lap_jaxpr(jaxpr, jsize,
                      tuple(nonzeros1), tuple(nonzeros2), tuple(instantiate))

@weakref_lru_cache
def _lap_jaxpr(jaxpr, jsize, nonzeros1, nonzeros2, instantiate):
    assert len(jaxpr.in_avals) == len(nonzeros1) == len(nonzeros2)
    f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
    f_jvp, out_nonzeros = f_lap_traceable(lap_fun(lap_subtrace(f), jsize, instantiate),
                                          nonzeros1, nonzeros2)
    jac_avals = [aval.update(shape=(jsize, *aval.shape))
                 for aval, nz in zip(jaxpr.in_avals, nonzeros2) if nz]
    lap_avals = [aval for aval, nz in zip(jaxpr.in_avals, nonzeros2) if nz]
    avals_in = [*jaxpr.in_avals, *jac_avals, *lap_avals]
    jaxpr_out, avals_out, literals_out = pe.trace_to_jaxpr_dynamic(f_jvp, avals_in)
    return core.ClosedJaxpr(jaxpr_out, literals_out), out_nonzeros()

@lu.transformation_with_aux
def f_lap_traceable(nonzeros1, nonzeros2, *primals_nzjacs_nzlaps):
    assert len(nonzeros1) == len(nonzeros2)
    num_primals = len(nonzeros1)
    primals = list(primals_nzjacs_nzlaps[:num_primals])
    nzjacs_nzlaps = iter(primals_nzjacs_nzlaps[num_primals:])
    jacs = [next(nzjacs_nzlaps) if nz else Zero.from_value(p)
            for p, nz in zip(primals, nonzeros1)]
    laps = [next(nzjacs_nzlaps) if nz else Zero.from_value(p)
            for p, nz in zip(primals, nonzeros2)]
    primals_out, jacs_out, laps_out = yield (primals, jacs, laps), {}
    out_nonzeros1 = [type(t) is not Zero for t in jacs_out]
    out_nonzeros2 = [type(t) is not Zero for t in laps_out]
    nonzero_jacs_out = [t for t in jacs_out if type(t) is not Zero]
    nonzero_laps_out = [t for t in laps_out if type(t) is not Zero]
    yield (list(primals_out) + nonzero_jacs_out + nonzero_laps_out,
           (out_nonzeros1, out_nonzeros2))


def _pjit_lap_rule(primals_in, jacs_in, laps_in,
              jaxpr, in_shardings, out_shardings,
              resource_env, donated_invars, name, keep_unused, inline):
    jsize, = set(map(lambda x: x.shape[0], tree_flatten(jacs_in)[0]))
    is_nz_jacs_in = [type(t) is not Zero for t in jacs_in]
    is_nz_laps_in = [type(t) is not Zero for t in laps_in]
    jaxpr_lap, (is_nz_jacs_out, is_nz_laps_out) = lap_jaxpr(
        jaxpr, jsize, is_nz_jacs_in, is_nz_laps_in, instantiate=False)

    def _filter_zeros(is_nz_l, l):
        return (x for nz, x in zip(is_nz_l, l) if nz)
    _fz_jacs_in = partial(_filter_zeros, is_nz_jacs_in)
    _fz_laps_in = partial(_filter_zeros, is_nz_laps_in)
    _fz_jacs_out = partial(_filter_zeros, is_nz_jacs_out)
    _fz_laps_out = partial(_filter_zeros, is_nz_laps_out)

    insd, outsd, dovar = in_shardings, out_shardings, donated_invars
    outputs = pjit_p.bind(
        *primals_in, *_fz_jacs_in(jacs_in), *_fz_laps_in(laps_in),
        jaxpr=jaxpr_lap,
        in_shardings=(*insd, *_fz_jacs_in(insd), *_fz_laps_in(insd)),
        out_shardings=(*outsd, *_fz_jacs_out(outsd), *_fz_laps_out(outsd)),
        resource_env=resource_env,
        donated_invars=(*dovar, *_fz_jacs_in(dovar), *_fz_laps_in(dovar)),
        name=name,
        keep_unused=keep_unused,
        inline=inline)

    primals_out, nzjacs_nzlaps = split_list(outputs, [len(jaxpr.jaxpr.outvars)])
    assert len(primals_out) == len(jaxpr.jaxpr.outvars)
    nzjacs_nzlaps_it = iter(nzjacs_nzlaps)
    jacs_out = [next(nzjacs_nzlaps_it) if nz else Zero(aval)
                for nz, aval in zip(is_nz_jacs_out, jaxpr.out_avals)]
    laps_out = [next(nzjacs_nzlaps_it) if nz else Zero(aval)
                for nz, aval in zip(is_nz_laps_out, jaxpr.out_avals)]
    return primals_out, jacs_out, laps_out

lap_rules[pjit_p] = _pjit_lap_rule
