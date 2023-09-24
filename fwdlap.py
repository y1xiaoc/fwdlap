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

from jax import lax
import jax.numpy as jnp
from jax.tree_util import (register_pytree_node, tree_structure,
                           treedef_is_leaf, tree_flatten, tree_unflatten,)

from jax import core
from jax.extend import linear_util as lu
from jax.util import unzip2
from jax.experimental.jet import jet_rules


def lap(fun, primals, series):
  try:
    order, = set(map(len, series))
  except ValueError:
    msg = "jet terms have inconsistent lengths for different arguments"
    raise ValueError(msg) from None

  # TODO(mattjj): consider supporting pytree inputs
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
  out_primals, out_terms = lap_fun(lap_subtrace(f), order).call_wrapped(primals, series)
  return tree_unflatten(out_tree(), out_primals), tree_unflatten(out_tree(), out_terms)

@lu.transformation
def lap_fun(order, primals, series):
  with core.new_main(LapTrace) as main:
    main.order = order
    out_primals, out_terms = yield (main, primals, series), {}
    del main
  out_terms = [[jnp.zeros_like(p)] * order if s is zero_series else s
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
    assert type(terms) in (ZeroSeries, list, tuple)
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
    order = self.main.order              # pytype: disable=attribute-error
    primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
    series_in = [[zero_term] * order if s is zero_series else s
                 for s in series_in]
    # TODO(mattjj): avoid always instantiating zeros
    series_in = [[jnp.zeros(np.shape(x), dtype=jnp.result_type(x))
                  if t is zero_term else t for t in series]
                 for x, series in zip(primals_in, series_in)]
    rule = jet_rules[primitive]
    primal_out, terms_out = rule(primals_in, series_in, **params)
    if not primitive.multiple_results:
      return LapTracer(self, primal_out, terms_out)
    else:
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
    # TODO(mattjj): don't just ignore custom jvp rules?
    del primitive, jvp  # Unused.
    return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
    del primitive, fwd, bwd, out_trees  # Unused.
    return fun.call_wrapped(*tracers)


class ZeroTerm: pass
zero_term = ZeroTerm()
register_pytree_node(ZeroTerm, lambda z: ((), None), lambda _, xs: zero_term)

class ZeroSeries: pass
zero_series = ZeroSeries()
register_pytree_node(ZeroSeries, lambda z: ((), None), lambda _, xs: zero_series)


call_param_updaters: dict[core.Primitive, Callable[..., Any]] = {}


### rule definitions

lap_rules = {}

