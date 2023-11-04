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
try:
    from jax.extend import linear_util as lu
except ImportError:
    from jax import linear_util as lu
from jax.util import split_list, safe_map as smap
from jax.api_util import flatten_fun_nokwargs
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters.ad import Zero

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
    f, out_tree = flatten_fun_output(lu.wrap_init(fun))
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
    f, f_out_tree = flatten_fun_output(lu.wrap_init(fun))
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
    in_tracers = smap(partial(LapTracer, trace), primals, jacobians, laplacians)
    ans = yield in_tracers, {}
    out_tracers = smap(trace.full_raise, ans)
    out_primals, out_jacs, out_laps = unzip3((t.primal, t.jacobian, t.laplacian)
                                             for t in out_tracers)
    yield out_primals, out_jacs, out_laps


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
        f_lap, out_tree_def = flatten_fun_nokwargs(lap_subtrace(f, self.main), in_tree_def)
        update_params = call_param_updaters.get(call_primitive)
        new_params = (update_params(params, len(primals_jacs_laps))
                      if update_params else params)
        result = call_primitive.bind(f_lap, *primals_jacs_laps, **new_params)
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
            return smap(partial(LapTracer, trace), primals, jacs, laps)
        return out, todo

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *,
                                symbolic_zeros):
        primals_in, jacs_in, laps_in = unzip3((t.primal, t.jacobian, t.laplacian)
                                              for t in tracers)
        primals_in = smap(core.full_lower, primals_in)
        primals_out, jacs_out, laps_out = vhv_by_jvp(
            wrap_custom_jvp(jvp).call_wrapped, primals_in, jacs_in, laps_in,
            inner_jvp=wrap_custom_jvp(_unwrap(jvp)).call_wrapped)
        return [LapTracer(self, p, j, l)
                for p, j, l in zip(primals_out, jacs_out, laps_out)]

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        raise TypeError("can't apply forward-mode laplacian to a custom_vjp "
                        "function.")


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


def _unwrap(wrapped: lu.WrappedFun) -> lu.WrappedFun:
    return lu.WrappedFun(wrapped.f, wrapped.transforms[1:],
                         wrapped.stores[1:], wrapped.params, None, None)


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
        o1 = jax.tree_map(zero_tangent_from_primal, o0)
        return o0, o1, o2_2
    o1, o2_1 = jax.vmap(vhv, in_axes=0, out_axes=0)(z1)
    _sum0 = lambda x: x.sum(0) if type(x) is not Zero else x
    _add_o2 = lambda a, b: ad.add_tangents(_sum0(a), b)
    o2 = smap(_add_o2, o2_1, o2_2) if multi_out else _add_o2(o2_1, o2_2)
    return o0, o1, o2


def primitive_by_jvp(primitive, primals_in, jacs_in, laps_in, **params):
    func = partial(primitive.bind, **params)
    f_jvp = partial(my_jvp, func)
    return vhv_by_jvp(f_jvp, primals_in, jacs_in, laps_in, inner_jvp=None)


### rule definitions

lap_rules: dict[core.Primitive, Callable[..., Any]] = {}


def defelemwise(prim, holomorphic=False):
    lap_rules[prim] = partial(elemwise_prop, prim, holomorphic)

def elemwise_prop(prim, holomorphic, primals_in, jacs_in, laps_in, **params):
    assert not prim.multiple_results
    pprim = partial(prim.bind, **params)
    z0, z1, z2 = primals_in, jacs_in, laps_in
    # check shape and dtype
    oinfo = jax.eval_shape(pprim, *primals_in)
    cplx_out = jnp.iscomplexobj(oinfo)
    has_cplx = any(jnp.iscomplexobj(z) for z in z0) or cplx_out
    if (any(z.shape != oinfo.shape for z in z0) or (has_cplx and not holomorphic)):
        return primitive_by_jvp(prim, primals_in, jacs_in, laps_in, **params)
    # calculations start
    o0, o2_2 = my_jvp(pprim, z0, z2)
    if all(type(j) is Zero for j in jacs_in):
        o1 = zero_tangent_from_primal(o0)
        return o0, o1, o2_2
    # now jacs are not all zero
    o1 = jax.vmap(lambda z: my_jvp(pprim, z0, z)[1], 0, 0)(z1)
    nonzero_idx = [i for i, jac in enumerate(z1) if type(jac) is not Zero]
    hess_fn = jax.hessian(pprim, argnums=nonzero_idx, holomorphic=cplx_out)
    for _ in range(o0.ndim):
        hess_fn = jax.vmap(hess_fn, in_axes=0)
    if cplx_out:
        z0 = smap(lambda z: z.astype(o0.dtype), z0)
    hess_mat = hess_fn(*z0)
    o2 = o2_2
    for i, nzi in enumerate(nonzero_idx):
        for j, nzj in enumerate(nonzero_idx[i:], i):
            hess = hess_mat[i][j]
            if i != j:
                hess *= 2
            o2 = ad.add_tangents(o2, hess * (z1[nzi] * z1[nzj]).sum(0))
    return o0, o1, o2

# currently do not enable any rules as benchmark results are unclear
# defelemwise(lax.exp_p)
# defelemwise(lax.exp2_p)
# defelemwise(lax.expm1_p)
# defelemwise(lax.log_p)
# defelemwise(lax.log1p_p)
# defelemwise(lax.logistic_p)
# defelemwise(lax.sin_p)
# defelemwise(lax.cos_p)
# defelemwise(lax.tan_p)
# defelemwise(lax.asin_p)
# defelemwise(lax.acos_p)
# defelemwise(lax.atan_p)
# defelemwise(lax.atan2_p)
# defelemwise(lax.sinh_p)
# defelemwise(lax.cosh_p)
# defelemwise(lax.tanh_p)
# defelemwise(lax.asinh_p)
# defelemwise(lax.acosh_p)
# defelemwise(lax.atanh_p)
# defelemwise(lax.sqrt_p)
# defelemwise(lax.rsqrt_p)
# defelemwise(lax.cbrt_p)
# defelemwise(lax.pow_p)
# defelemwise(lax.integer_pow_p)
# defelemwise(lax.div_p)
# defelemwise(lax.lgamma_p)
# defelemwise(lax.digamma_p)
# defelemwise(lax.polygamma_p)
# defelemwise(lax.igamma_p)
# defelemwise(lax.igammac_p)
# defelemwise(lax.bessel_i0e_p)
# defelemwise(lax.bessel_i1e_p)
# defelemwise(lax.erf_p)
# defelemwise(lax.erfc_p)
# defelemwise(lax.erf_inv_p)


def defmultivar(prim):
    lap_rules[prim] = partial(multivar_prop, prim)

def multivar_prop(prim, primals_in, jacs_in, laps_in, **params):
    pprim = partial(prim.bind, **params)
    z0, z1, z2 = primals_in, jacs_in, laps_in
    o0, o2_2 = my_jvp(pprim, z0, z2)
    if all(type(j) is Zero for j in jacs_in):
        o1 = zero_tangent_from_primal(o0)
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
        triu_z1 = [zero_tangent_from_primal(p) if j <= i else t
                   for j, (p, t) in enumerate(zip(z0,z1))]
        diag_z1 = [zero_tangent_from_primal(p) if j != i else t
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
                 for aval, nz in zip(jaxpr.in_avals, nonzeros1) if nz]
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
