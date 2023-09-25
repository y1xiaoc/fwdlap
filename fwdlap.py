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
from jax.tree_util import (register_pytree_node, tree_structure, tree_map,
                           treedef_is_leaf, tree_flatten, tree_unflatten,)

from jax import core
from jax.dtypes import float0
try:
    from jax.extend import linear_util as lu
except ImportError:
    from jax import linear_util as lu
from jax.util import unzip2, split_list, safe_map as smap, safe_zip as szip
from jax.interpreters.ad import replace_float0s

from jax.experimental import jet
from jax.experimental.jet import jet_rules, zero_term, zero_series
jet.fact = lambda n: jax.lax.prod(range(1, n + 1)) # modification from @YouJiacheng


def lap(fun, primals, series):
    if set(map(len, series)) != {2}:
        msg = "jet terms have inconsistent lengths for different arguments"
        raise ValueError(msg)

    for i, (x, terms) in enumerate(zip(primals, series)):
        treedef = tree_structure(x)
        if not treedef_is_leaf(treedef):
            raise ValueError(f"primal value at position {i} is not an array")
        for j, t in enumerate(terms):
            treedef = tree_structure(t)
            if not treedef_is_leaf(treedef):
                raise ValueError(f"term {j} for argument {i} is not an array")

    @lu.transformation_with_aux
    def flatten_fun_output(*args):
        ans = yield args, {}
        yield tree_flatten(ans)

    f, out_tree = flatten_fun_output(lu.wrap_init(fun))
    out_primals, out_terms = lap_fun(lap_subtrace(f)).call_wrapped(primals, series)
    return tree_unflatten(out_tree(), out_primals), tree_unflatten(out_tree(), out_terms)

@lu.transformation
def lap_fun(primals, series):
    with core.new_main(LapTrace) as main:
        out_primals, out_terms = yield (main, primals, series), {}
        del main
    out_terms = [[jnp.zeros_like(p)[None], jnp.zeros_like(p)] if s is zero_series else s
                             for p, s in zip(out_primals, out_terms)]
    yield out_primals, out_terms

@lu.transformation
def lap_subtrace(main, primals, series):
    trace = LapTrace(main, core.cur_sublevel())
    in_tracers = map(partial(LapTracer, trace), primals, series)
    ans = yield in_tracers, {}
    out_tracers = map(trace.full_raise, ans)
    out_primals, out_terms = unzip2((t.primal, t.terms) for t in out_tracers)
    yield out_primals, out_terms

@lu.transformation_with_aux
def traceable(in_tree_def, *primals_and_series):
    primals_in, series_in = tree_unflatten(in_tree_def, primals_and_series)
    primals_out, series_out = yield (primals_in, series_in), {}
    out_flat, out_tree_def = tree_flatten((primals_out, series_out))
    yield out_flat, out_tree_def


class LapTracer(core.Tracer):
    __slots__ = ["primal", "terms"]

    def __init__(self, trace, primal, terms):
        assert type(terms) in (jet.ZeroSeries, list, tuple)
        self._trace = trace
        self.primal = primal
        self.terms = terms

    @property
    def aval(self):
        return core.get_aval(self.primal)

    def full_lower(self):
        if self.terms is zero_series or all(t is zero_term for t in self.terms):
            return core.full_lower(self.primal)
        else:
            return self

class LapTrace(core.Trace):

    def pure(self, val):
        return LapTracer(self, val, zero_series)

    def lift(self, val):
        return LapTracer(self, val, zero_series)

    def sublift(self, val):
        return LapTracer(self, val.primal, val.terms)

    def process_primitive(self, primitive, tracers, params):
        primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
        jacs_in, laps_in = align_input_series(primals_in, series_in)
        if primitive in lap_rules:
            rule = lap_rules[primitive]
            primal_out, jac_out, lap_out = rule(
                primals_in, jacs_in, laps_in, **params)
        elif primitive in jet_rules:
            primal_out, jac_out, lap_out = primitive_by_jet(
                primitive, primals_in, jacs_in, laps_in, **params)
        else:
            primal_out, jac_out, lap_out = primitive_by_jvp(
                primitive, primals_in, jacs_in, laps_in, **params)
        if not primitive.multiple_results:
            return LapTracer(self, primal_out, (jac_out, lap_out))
        else:
            terms_out = szip(jac_out, lap_out)
            return [LapTracer(self, p, ts) for p, ts in zip(primal_out, terms_out)]

    def process_call(self, call_primitive, f, tracers, params):
        primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
        primals_and_series, in_tree_def = tree_flatten((primals_in, series_in))
        f_jet, out_tree_def = traceable(lap_subtrace(f, self.main), in_tree_def)
        update_params = call_param_updaters.get(call_primitive)
        new_params = (update_params(params, len(primals_and_series))
                      if update_params else params)
        result = call_primitive.bind(f_jet, *primals_and_series, **new_params)
        primals_out, series_out = tree_unflatten(out_tree_def(), result)
        return [LapTracer(self, p, ts) for p, ts in zip(primals_out, series_out)]

    def post_process_call(self, call_primitive, out_tracers, params):
        primals, series = unzip2((t.primal, t.terms) for t in out_tracers)
        out, treedef = tree_flatten((primals, series))
        del primals, series
        main = self.main
        def todo(x):
            primals, series = tree_unflatten(treedef, x)
            trace = LapTrace(main, core.cur_sublevel())
            return map(partial(LapTracer, trace), primals, series)
        return out, todo

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *,
                                symbolic_zeros):
        primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
        jacs_in, laps_in = align_input_series(primals_in, series_in)
        primals_in = smap(core.full_lower, primals_in)
        def wrap_jvp(jvp):
            def wrapped(p_in, t_in):
                outs = jvp(*p_in, *t_in)
                return split_list(outs, [len(outs) // 2])
            return wrapped
        primal_out, jac_out, lap_out = hvv_by_jvp(
            wrap_jvp(jvp.call_wrapped), primals_in, jacs_in, laps_in,
            inner_jvp=wrap_jvp(jvp.f))
        terms_out = szip(jac_out, lap_out)
        return [LapTracer(self, p, ts) for p, ts in zip(primal_out, terms_out)]

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        del primitive, fwd, bwd, out_trees  # Unused.
        return fun.call_wrapped(*tracers)


call_param_updaters: dict[core.Primitive, Callable[..., Any]] = {}


def align_input_series(primals_in, series_in):
    series_in = [[zero_term] * 2 if s is zero_series else s
                 for s in series_in]
    jacs_in, laps_in = unzip2(series_in)
    try:
        n_coord, = set(j.shape[0] for j in jacs_in if j is not zero_term)
    except ValueError:
        msg = "jacobians have inconsistent 1st dimension for different arguments"
        raise ValueError(msg) from None
    jacs_in = [jnp.zeros((n_coord, *np.shape(x)), dtype=jnp.result_type(x))
               if t is zero_term else t  for x, t in zip(primals_in, jacs_in)]
    laps_in = [jnp.zeros(np.shape(x), dtype=jnp.result_type(x))
               if t is zero_term else t  for x, t in zip(primals_in, laps_in)]
    return jacs_in, laps_in


def hvv_by_jvp(f_jvp, primals_in, jacs_in, laps_in, inner_jvp=None):
    z0 = primals_in
    z1 = tree_map(recast_np_float0, primals_in, jacs_in)
    z2 = tree_map(recast_np_float0, primals_in, laps_in)
    if inner_jvp is None:
        inner_jvp = f_jvp
    def hvv(v):
        inner = lambda *a: inner_jvp(a, v)
        (_, o1), (_, o2_1) = jax.jvp(inner, z0, v)
        return o1, o2_1
    o1, o2_1 = jax.vmap(hvv, in_axes=0, out_axes=0)(z1)
    o0, o2_2 = f_jvp(z0, z2)
    cast_f0s = lambda term: jax.tree_map(replace_float0s, o0, term)
    o1, o2_1 = smap(jax.vmap(cast_f0s), (o1, o2_1))
    o2_2 = cast_f0s(o2_2)
    o2 = jax.tree_map(lambda a, b: a.sum(0) + b, o2_1, o2_2)
    return o0, o1, o2


def primitive_by_jvp(primitive, primals_in, jacs_in, laps_in, **params):
    func = partial(primitive.bind, **params)
    f_jvp = partial(jax.jvp, func)
    return hvv_by_jvp(f_jvp, primals_in, jacs_in, laps_in, inner_jvp=None)


def primitive_by_jet(primitive, primals_in, jacs_in, laps_in, **params):
    rule = jet_rules[primitive]
    n_coord, = set(map(lambda j: j.shape[0], jacs_in))
    z0, z1 = primals_in, jacs_in
    z2 = jax.tree_map(lambda a: a / n_coord, laps_in)
    def jet_call(p_in, j_in, l_in):
        s_in = list(map(list, zip(j_in, l_in)))
        p_out, s_out = rule(p_in, s_in, **params)
        j_out, l_out = unzip2(s_out) if primitive.multiple_results else s_out
        return p_out, j_out, l_out
    o0, o1, o2 = jax.vmap(jet_call, (None, 0, None), (None, 0, 0))(z0, z1, z2)
    o2 = jax.tree_map(lambda a: a.sum(0), o2)
    return o0, o1, o2


def recast_np_float0(primal, tangent):
    if core.primal_dtype_to_tangent_dtype(jnp.result_type(primal)) == float0:
        return np.zeros(tangent.shape, dtype=float0)
    else:
        return tangent


### rule definitions

lap_rules = {}

