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
# This code takes references from jet and jvp in jax

from typing import Any, Callable

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import (tree_structure, treedef_is_leaf,
                           tree_flatten, tree_unflatten, Partial)

from jax import core
from jax.extend import core as ext_core
try:
    from jax.extend import linear_util as lu
except ImportError:
    from jax import linear_util as lu
from jax.util import split_list, safe_map as smap
from jax.api_util import flatten_fun_nokwargs, shaped_abstractify, debug_info
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters.ad import Zero, instantiate_zeros
from jax.dtypes import float0

from jax._src.util import unzip3, weakref_lru_cache
from jax.experimental.pjit import pjit_p


def lap(fun, primals, jacobians, laplacians):
    """
    Computes the (forward mode) jacobian and laplacian of a function `fun`.

    This function has very similar semantics to `jax.jvp`, except that it
    requires batched tangent vectors (jacobians) and laplacians for each input,
    and returns batched jvp and the cumulated laplacian from the batched tangents.

    Args:
        fun: A function that takes in `primals` and returns an output.
          Its arguments have to be arrays or scalars, but not in nested python
          containers. Its output can be any pytrees of arrays or scalars.
        primals: The primal values at which the jacobian of `fun` should be
          evaluated. Should be either a tuple or a list of arguments. and its
          length should be equal to the number of positional parameters of `fun`.
        jacobians: The jacobian matrices (batched tangent vectors) for each
          input to evaluate the jvp. Should be either a tuple or a list of
          arguments with the same tree structure as `primals`, with an exception
          of symbolic `Zero` values that represent zero jacobians. The jacobians
          should have an extra leading dimension compared to the primal values,
          which is the batch size and will be summed over in the laplacian.
        laplacians: The laplacian vectors for each input to evaluate the
          forward laplacian. Should be either a tuple or a list of arguments
          with the same tree structure as `primals`, with an exception of
          symbolic `Zero` values that represent zero laplacians.

    Returns:
        A tuple of three elements:
        - The outputs of `fun` at `primals`.
        - Jacobian matrices with respect to each output.
        - Laplacian vectors with respect to each output.
    """
    check_no_nested(primals, jacobians, laplacians)
    jsize = get_jsize(jacobians)
    f, out_tree = flatten_fun_output(
        lu.wrap_init(fun, debug_info=debug_info("lap", fun, primals, {})))
    out_primals, out_jacs, out_laps = lap_fun(
        lap_subtrace(f), jsize, True).call_wrapped(
        primals, jacobians, laplacians)
    return (tree_unflatten(out_tree(), out_primals),
            tree_unflatten(out_tree(), out_jacs),
            tree_unflatten(out_tree(), out_laps))


def lap_partial(fun, primals, example_jacs, example_laps):
    """
    The partial eval version of `lap`.

    This function will compute the primal output of `fun` and postpone
    the jacobian and laplacian calculation in a returned function.
    It takes exact same arguments as `lap`, but this time `example_jacs`
    and `example_laps` are only used to determine the shape.

    Args:
        fun: A function that takes in `primals` and returns an output.
          Its arguments have to be arrays or scalars, but not in nested python
          containers. Its output can be any pytrees of arrays or scalars.
        primals: The primal values at which the jacobian of `fun` should be
          evaluated. Should be either a tuple or a list of arguments. and its
          length should be equal to the number of positional parameters of `fun`.
        example_jacs: The jacobian matrices (batched tangent vectors) for each
          input to evaluate the jvp. See `lap` for more details. The value does
          not matter, only the shape (or whether it's symboilc `Zero`) is used.
        example_laps: The laplacian vectors for each input to evaluate the
          forward laplacian. See `lap` for more details. Only the shape
          (or whether it's symboilc `Zero`) is used.

    Returns:
        A tuple of two elements:
        - The output of `fun` at `primals`.
        - A function that takes in the jacobian and laplacian arguments
          and returns the jacobian and laplacian of the output. The tree
          structure of jacobian and laplatian arguments should be the same
          as `example_jacs` and `example_laps` respectively.
    """
    # make the lap tracer with wrapped (flattened) function
    check_no_nested(primals, example_jacs, example_laps)
    jsize = get_jsize(example_jacs)
    f, f_out_tree = flatten_fun_output(
        lu.wrap_init(fun, debug_info=debug_info("lap_partial", fun, primals, {})))
    f_lap = lap_fun(lap_subtrace(f), jsize, True)
    # partial eval, including pre and post process
    in_pvals = (tuple(pe.PartialVal.known(p) for p in primals)
                + tuple(pe.PartialVal.unknown(core.get_aval(p))
                        for p in tree_flatten((example_jacs, example_laps))[0]))
    _, in_tree = tree_flatten((primals, example_jacs, example_laps))
    f_lap_flat, lap_out_tree = flatten_fun_nokwargs(f_lap, in_tree)
    jaxpr, out_pvals, consts = pe.trace_to_jaxpr_nounits(f_lap_flat, in_pvals)
    op_pvals, oj_pvals, ol_pvals = tree_unflatten(lap_out_tree(), out_pvals)
    # collect known primals out
    f_out_tree = f_out_tree()
    assert all(opp.is_known() for opp in op_pvals)
    op_flat = [opp.get_known() for opp in op_pvals]
    primals_out = tree_unflatten(f_out_tree, op_flat)
    # build function for unknown laplacian
    def lap_pe(consts, jacs, laps):
        oj_ol_flat = core.eval_jaxpr(jaxpr, consts, *tree_flatten((jacs, laps))[0])
        oj_ol_flat_ = iter(oj_ol_flat)
        oj_flat = [ojp.get_known() if ojp.is_known() else next(oj_ol_flat_)
                   for ojp in oj_pvals]
        ol_flat = [olp.get_known() if olp.is_known() else next(oj_ol_flat_)
                   for olp in ol_pvals]
        assert next(oj_ol_flat_, None) is None
        return (tree_unflatten(f_out_tree, oj_flat),
                tree_unflatten(f_out_tree, ol_flat))
    # make partial eval a pytree
    return primals_out, Partial(lap_pe, consts)


def get_jsize(jacobians):
    try:
        jsize, = set(map(lambda x:x.shape[0], tree_flatten(jacobians)[0]))
        return jsize
    except ValueError:
        msg = "jacobians have inconsistent first dimensions for different arguments"
        raise ValueError(msg) from None


def check_no_nested(primals, jacobians, laplacians):
    for i, (x, j, l) in enumerate(zip(primals, jacobians, laplacians)):
        for t, name in ((x, "primal"), (j, "jacobian"), (l, "laplacian")):
            treedef = tree_structure(t)
            if not treedef_is_leaf(treedef):
                raise ValueError(f"{name} value at position {i} is not an array")


@lu.transformation2
def lap_fun(f, jsize, instantiate, primals, jacobians, laplacians):
    tag = core.TraceTag()
    jacobians = [zero_tangent_from_primal(j, jsize) if type(j) is not Zero
                 and lax.dtype(j) == float0 else j for j in jacobians]
    laplacians = [zero_tangent_from_primal(l, jsize) if type(l) is not Zero
                  and lax.dtype(l) == float0 else l for l in laplacians]
    out_primals, out_jacs, out_laps = f(tag, jsize, primals, jacobians, laplacians)
    if type(instantiate) is bool:
        instantiate = [instantiate] * len(out_jacs)
    out_jacs = [instantiate_zeros(j) if inst else j
                for j, inst in zip(out_jacs, instantiate)]
    out_laps = [instantiate_zeros(l) if inst else l
                for l, inst in zip(out_laps, instantiate)]
    return out_primals, out_jacs, out_laps


@lu.transformation2
def lap_subtrace(f, tag, jsize, primals, jacobians, laplacians):
    with core.take_current_trace() as parent_trace:
        trace = LapTrace(tag, parent_trace, jsize)
        in_tracers = smap(partial(maybe_lap_tracer, trace),
                          primals, jacobians, laplacians)
        with core.set_current_trace(trace):
            ans = f(*in_tracers)
        out_primals, out_jacs, out_laps = unzip3(smap(trace.to_pjl_tuple, ans))
    return out_primals, out_jacs, out_laps


def maybe_lap_tracer(trace, primal, jacobian, laplacian):
    if ((type(jacobian) is Zero or lax.dtype(jacobian) == float0)
      and (type(laplacian) is Zero or lax.dtype(laplacian) == float0)):
        return primal
    else:
        return LapTracer(trace, primal, jacobian, laplacian)


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

    def __init__(self, tag, parent_trace, jsize):
        self.tag = tag
        self.parent_trace = parent_trace
        self.jsize = jsize

    def to_pjl_tuple(self, val):
        if isinstance(val, LapTracer) and val._trace.tag is self.tag:
            return val.primal, val.jacobian, val.laplacian
        else:
            return (val,
                    zero_tangent_from_primal(val, self.jsize),
                    zero_tangent_from_primal(val))

    def process_primitive(self, primitive, tracers, params):
        jsize = self.jsize
        primals_in, jacs_in, laps_in = unzip3(smap(self.to_pjl_tuple, tracers))
        with core.set_current_trace(self.parent_trace):
            if primitive in lap_rules:
                rule = lap_rules[primitive]
                primal_out, jac_out, lap_out = rule(
                    jsize, primals_in, jacs_in, laps_in, **params)
            else:
                primal_out, jac_out, lap_out = primitive_by_jvp(
                    primitive, jsize, primals_in, jacs_in, laps_in, **params)
        if not primitive.multiple_results:
            return maybe_lap_tracer(self, primal_out, jac_out, lap_out)
        else:
            return [maybe_lap_tracer(self, p, j, l)
                    for p, j, l in zip(primal_out, jac_out, lap_out)]

    def process_custom_jvp_call(self, prim, fun, jvp, tracers, *,
                                symbolic_zeros):
        primals_in, jacs_in, laps_in = unzip3(smap(self.to_pjl_tuple, tracers))
        if all(type(t.jacobian) is type(t.laplacian) is Zero for t in tracers):
            return prim.bind_with_trace(self.parent_trace, (fun, jvp, *primals_in),
                                  dict(symbolic_zeros=symbolic_zeros))
        if symbolic_zeros:
            raise NotImplementedError("symbolic_zeros not implemented")
        with core.set_current_trace(self.parent_trace):
            jacs_in = smap(ad.instantiate_zeros, jacs_in)
            laps_in = smap(ad.instantiate_zeros, laps_in)
            in_avals = smap(shaped_abstractify, (*primals_in, *laps_in))
            jaxpr, _, consts, *_ = pe.trace_to_jaxpr_dynamic(jvp, in_avals)
            def _jvp(p_in, t_in):
                outs = core.eval_jaxpr(jaxpr, consts, *p_in, *t_in)
                p_out, t_out = split_list(outs, [len(outs) // 2])
                return p_out, t_out
            primals_out, jacs_out, laps_out = vhv_by_jvp(
                _jvp, self.jsize, primals_in, jacs_in, laps_in)
        return [maybe_lap_tracer(self, p, j, l)
                for p, j, l in zip(primals_out, jacs_out, laps_out)]

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        raise TypeError("can't apply forward-mode laplacian to a custom_vjp "
                        "function.")


call_param_updaters: dict[ext_core.Primitive, Callable[..., Any]] = {}


def zero_tangent_from_primal(primal, jsize=None):
    zero = Zero.from_primal_value(primal)
    if jsize is None:
        return zero
    aval = zero.aval
    return Zero(aval.update(shape=(jsize, *aval.shape)))


@lu.transformation_with_aux2
def flatten_fun_output(f, store, *args):
    ans = f(*args)
    ans, tree = tree_flatten(ans)
    store.store(tree)
    return ans


def my_jvp(fun, primals, tangents):
    # this jvp is transparant to Zero, and assumes flattened input
    f, out_tree = flatten_fun_output(
        lu.wrap_init(fun, debug_info=debug_info("lap_innerjvp", fun, primals, {})))
    jvp_f = ad.jvp(f, instantiate=False)
    out_primals, out_tangents = jvp_f.call_wrapped(primals, tangents)
    out_tree = out_tree()
    return (tree_unflatten(out_tree, out_primals),
            tree_unflatten(out_tree, out_tangents))


def vhv_by_jvp(f_jvp, jsize, primals_in, jacs_in, laps_in):
    z0, z1, z2 = primals_in, jacs_in, laps_in
    def vhv(v):
        inner = lambda *a: f_jvp(a, v)[1]
        return my_jvp(inner, z0, v)
    # second term in laplacian
    o0, o2_2 = f_jvp(z0, z2)
    multi_out = not treedef_is_leaf(tree_structure(o0))
    # jacobian and first term in laplacian, handle all empty case
    if all(type(j) is Zero for j in z1):
        zero_o1_fn = partial(zero_tangent_from_primal, jsize=jsize)
        o1 = jax.tree_util.tree_map(zero_o1_fn, o0)
        return o0, o1, o2_2
    o1, o2_1 = jax.vmap(vhv, in_axes=0, out_axes=0)(z1)
    _sum0 = lambda x: x.sum(0) if type(x) is not Zero else x
    _add_o2 = lambda a, b: ad.add_tangents(_sum0(a), b)
    o2 = smap(_add_o2, o2_1, o2_2) if multi_out else _add_o2(o2_1, o2_2)
    return o0, o1, o2


def primitive_by_jvp(primitive, jsize, primals_in, jacs_in, laps_in, **params):
    func = partial(primitive.bind, **params)
    f_jvp = partial(my_jvp, func)
    return vhv_by_jvp(f_jvp, jsize, primals_in, jacs_in, laps_in)


### rule definitions

lap_rules: dict[ext_core.Primitive, Callable[..., Any]] = {}


def defmultivar(prim):
    lap_rules[prim] = partial(multivar_prop, prim)

def multivar_prop(prim, jsize, primals_in, jacs_in, laps_in, **params):
    pprim = partial(prim.bind, **params)
    z0, z1, z2 = primals_in, jacs_in, laps_in
    o0, o2_2 = my_jvp(pprim, z0, z2)
    if all(type(j) is Zero for j in jacs_in):
        o1 = zero_tangent_from_primal(o0, jsize)
        return o0, o1, o2_2
    o1 = jax.vmap(lambda v: my_jvp(pprim, z0, v)[1], 0, 0)(z1)
    _mul2 = lambda x: 2*x if type(x) is not Zero else x
    _sum0 = lambda x: x.sum(0) if type(x) is not Zero else x
    def vhv(v1, v2):
        inner = lambda *a: my_jvp(pprim, a, v1)[1]
        return my_jvp(inner, z0, v2)[1]
    def vmapped_vhv(v1, v2):
        if not tree_flatten((v1, v2))[0]: # empty tree
            return zero_tangent_from_primal(o0)
        return jax.vmap(vhv, in_axes=0, out_axes=0)(v1, v2)
    o2 = o2_2
    for i in range(len(primals_in)):
        triu_z1 = [zero_tangent_from_primal(p, jsize) if j <= i else t
                   for j, (p, t) in enumerate(zip(z0,z1))]
        diag_z1 = [zero_tangent_from_primal(p, jsize) if j != i else t
                   for j, (p, t) in enumerate(zip(z0,z1))]
        o2_1_diag = vmapped_vhv(diag_z1, diag_z1)
        o2 = ad.add_tangents(_sum0(o2_1_diag), o2)
        o2_1_triu = vmapped_vhv(triu_z1, diag_z1)
        o2 = ad.add_tangents(_mul2(_sum0(o2_1_triu)), o2)
    return o0, o1, o2

defmultivar(lax.mul_p)
defmultivar(lax.dot_general_p)
defmultivar(lax.conv_general_dilated_p)
# This rule will only be faster when the operator is bilinear.
# Because the diagonal part of o2_1 is Zero.
# Hence we do not apply it for the following primitives.
# defmultivar(lax.div_p)
# defmultivar(lax.rem_p)
# defmultivar(lax.pow_p)
# defmultivar(lax.atan2_p)


def lap_jaxpr(jaxpr, jsize, nonzeros1, nonzeros2, instantiate):
    if type(instantiate) is bool:
        instantiate = (instantiate,) * len(jaxpr.out_avals)
    return _lap_jaxpr(jaxpr, jsize,
                      tuple(nonzeros1), tuple(nonzeros2), tuple(instantiate))

@weakref_lru_cache
def _lap_jaxpr(jaxpr, jsize, nonzeros1, nonzeros2, instantiate):
    assert len(jaxpr.in_avals) == len(nonzeros1) == len(nonzeros2)
    f = lu.wrap_init(ext_core.jaxpr_as_fun(jaxpr), debug_info=jaxpr.jaxpr.debug_info)
    f_jvp, out_nonzeros = f_lap_traceable(lap_fun(lap_subtrace(f), jsize, instantiate),
                                          nonzeros1, nonzeros2)
    jac_avals = [aval.update(shape=(jsize, *aval.shape))
                 for aval, nz in zip(jaxpr.in_avals, nonzeros1) if nz]
    lap_avals = [aval for aval, nz in zip(jaxpr.in_avals, nonzeros2) if nz]
    avals_in = [*jaxpr.in_avals, *jac_avals, *lap_avals]
    jaxpr_out, avals_out, literals_out = pe.trace_to_jaxpr_dynamic(f_jvp, avals_in)
    return ext_core.ClosedJaxpr(jaxpr_out, literals_out), out_nonzeros()

@lu.transformation_with_aux2
def f_lap_traceable(f, store, nonzeros1, nonzeros2, *primals_nzjacs_nzlaps):
    assert len(nonzeros1) == len(nonzeros2)
    num_primals = len(nonzeros1)
    primals = list(primals_nzjacs_nzlaps[:num_primals])
    nzjacs_nzlaps = iter(primals_nzjacs_nzlaps[num_primals:])
    jacs = [next(nzjacs_nzlaps) if nz else zero_tangent_from_primal(p)
            for p, nz in zip(primals, nonzeros1)]
    laps = [next(nzjacs_nzlaps) if nz else zero_tangent_from_primal(p)
            for p, nz in zip(primals, nonzeros2)]
    primals_out, jacs_out, laps_out = f(primals, jacs, laps)
    out_nonzeros1 = [type(t) is not Zero for t in jacs_out]
    out_nonzeros2 = [type(t) is not Zero for t in laps_out]
    nonzero_jacs_out = [t for t in jacs_out if type(t) is not Zero]
    nonzero_laps_out = [t for t in laps_out if type(t) is not Zero]
    store.store((out_nonzeros1, out_nonzeros2))
    return list(primals_out) + nonzero_jacs_out + nonzero_laps_out


def _pjit_lap_rule(jsize, primals_in, jacs_in, laps_in, *, jaxpr, **params):
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

    insd, outsd = params["in_shardings"], params["out_shardings"]
    dovar = params["donated_invars"]
    new_params = {
        **params,
        "jaxpr": jaxpr_lap,
        "in_shardings": (*insd, *_fz_jacs_in(insd), *_fz_laps_in(insd)),
        "out_shardings": (*outsd, *_fz_jacs_out(outsd), *_fz_laps_out(outsd)),
        "donated_invars": (*dovar, *_fz_jacs_in(dovar), *_fz_laps_in(dovar)),
    }
    if "in_layouts" in params:
        inlo, outlo = params["in_layouts"], params["out_layouts"]
        new_params["in_layouts"] = (*inlo, *_fz_jacs_in(inlo), *_fz_laps_in(inlo))
        new_params["out_layouts"] = (*outlo, *_fz_jacs_out(outlo), *_fz_laps_out(outlo))

    outputs = pjit_p.bind(
        *primals_in,
        *_fz_jacs_in(jacs_in),
        *_fz_laps_in(laps_in),
        **new_params
    )

    primals_out, nzjacs_nzlaps = split_list(outputs, [len(jaxpr.jaxpr.outvars)])
    assert len(primals_out) == len(jaxpr.jaxpr.outvars)
    nzjacs_nzlaps_it = iter(nzjacs_nzlaps)
    jacs_out = [next(nzjacs_nzlaps_it) if nz else Zero(aval)
                for nz, aval in zip(is_nz_jacs_out, jaxpr.out_avals)]
    laps_out = [next(nzjacs_nzlaps_it) if nz else Zero(aval)
                for nz, aval in zip(is_nz_laps_out, jaxpr.out_avals)]
    return primals_out, jacs_out, laps_out

lap_rules[pjit_p] = _pjit_lap_rule
