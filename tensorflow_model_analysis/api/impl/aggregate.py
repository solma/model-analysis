# Copyright 2018 Google LLC
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
"""Public API for performing evaluations using the EvalMetricsGraph."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy


import apache_beam as beam
import numpy as np

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import Any, Dict, Generator, Iterable, List, Optional, Text, Tuple

# For use in Beam type annotations, because Beam's support for Python types
# in Beam type annotations is not complete.
_BeamSliceKeyType = beam.typehints.Tuple[  # pylint: disable=invalid-name
    beam.typehints.Tuple[Text, beam.typehints.Union[bytes, int, float]], Ellipsis]

SAMPLE_ID = '__sample_id'


@beam.ptransform_fn
@beam.typehints.with_input_types(
    beam.typehints.Tuple[_BeamSliceKeyType, types.FeaturesPredictionsLabels])
@beam.typehints.with_output_types(
    beam.typehints.Tuple[_BeamSliceKeyType, beam.typehints.List[beam.typehints
                                                                .Any]])
def ComputePerSliceMetrics(  # pylint: disable=invalid-name
    slice_result,
    eval_shared_model,
    desired_batch_size = None,
    num_bootstrap_samples = 1,
):
  """PTransform for computing, aggregating and combining metrics.

  Args:
    slice_result: Incoming PCollection consisting of slice key and example.
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional batch size for batching in Aggregate.
    num_bootstrap_samples: Number of replicas to use in calculating uncertainty
      using bootstrapping.  If 1 is provided (default), aggregate metrics will
      be calculated with no uncertainty. If num_bootstrap_samples is > 0,
      multiple samples of each slice will be calculated using the Poisson
      bootstrap method. To calculate standard errors, num_bootstrap_samples
      should be 20 or more in order to provide useful data. More is better, but
      you pay a performance cost.

  Returns:
    DoOutputsTuple. The tuple entries are
    PCollection of (slice key, metrics) and
    PCollection of (slice key, plot metrics).
  """
  compute_with_sampling = False
  if num_bootstrap_samples < 1:
    raise ValueError(
        'num_bootstrap_samples should be > 0, got %d' % num_bootstrap_samples)
  if num_bootstrap_samples > 1:
    slice_result = slice_result | 'FanoutBootstrap' >> beam.ParDo(
        _FanoutBootstrapFn(num_bootstrap_samples))
    compute_with_sampling = True
  return (
      slice_result
      | 'CombinePerSlice' >> beam.CombinePerKey(
          _AggregateCombineFn(
              eval_shared_model=eval_shared_model,
              desired_batch_size=desired_batch_size,
              compute_with_sampling=compute_with_sampling))
      # Explicitly specify a fanout to alleviate memory issues.
      .with_hot_key_fanout(fanout=16)
      | 'InterpretOutput' >> beam.ParDo(
          _ExtractOutputDoFn(eval_shared_model=eval_shared_model)).with_outputs(
              _ExtractOutputDoFn.OUTPUT_TAG_PLOTS,
              main=_ExtractOutputDoFn.OUTPUT_TAG_METRICS))


class _FanoutBootstrapFn(beam.DoFn):
  """For each bootstrap sample you want, we fan out an additional slice."""

  def __init__(self, num_bootstrap_samples):
    self._num_bootstrap_samples = num_bootstrap_samples

  def process(self, element):
    slice_key, value = element
    for i in range(0, self._num_bootstrap_samples):
      # Make a copy of the slice key, prepend the sample ID.
      augmented_slice_key = (SAMPLE_ID, i) + copy.copy(slice_key)
      # This fans out the pipeline, but because we are reducing in a
      # CombinePerKey, we shouldn't have to deal with a great increase in
      # network traffic.
      yield (augmented_slice_key, value)


def _add_metric_variables(  # pylint: disable=invalid-name
    left,
    right):
  """Returns left and right metric variables combined."""
  if left is not None and right is not None:
    if len(left) != len(right):
      raise ValueError('metric variables lengths should match, but got '
                       '%d and %d' % (len(left), len(right)))
    return [x + y for x, y in zip(left, right)]
  elif left is not None:
    return left
  else:
    return right


class _AggState(object):
  """Combine state for AggregateCombineFn.

  There are two parts to the state: the metric variables (the actual state),
  and a list of FeaturesPredictionsLabels. See _AggregateCombineFn for why
  we need this.
  """

  def __init__(self):
    self.metric_variables = None  # type: Optional[types.MetricVariablesType]
    self.fpls = []  # type: List[types.FeaturesPredictionsLabels]

  def copy_from(  # pylint: disable=invalid-name
      self, other):
    if other.metric_variables:
      self.metric_variables = other.metric_variables
    self.fpls = other.fpls

  def __iadd__(self, other):
    self.metric_variables = _add_metric_variables(self.metric_variables,
                                                  other.metric_variables)
    self.fpls.extend(other.fpls)
    return self

  def add_fpl(  # pylint: disable=invalid-name
      self, fpl):
    self.fpls.append(fpl)

  def add_metrics_variables(  # pylint: disable=invalid-name
      self, metric_variables):
    self.metric_variables = _add_metric_variables(self.metric_variables,
                                                  metric_variables)


@beam.typehints.with_input_types(types.FeaturesPredictionsLabels)
@beam.typehints.with_output_types(beam.typehints.List[beam.typehints.Any])
class _AggregateCombineFn(beam.CombineFn):
  """Aggregate combine function.

  This function really does three things:
    1. Batching of FeaturesPredictionsLabels.
    3. "Partial reduction" of these batches by sending this through the
       "intro metrics" step.
    3. The "normal" combining of MetricVariables.

  What we really want to do is conceptually the following:
  Predictions | GroupByKey() | KeyAwareBatchElements()
              | ParDo(IntroMetrics()) | CombineValues(CombineMetricVariables()).

  but there's no way to KeyAwareBatchElements in Beam, and no way to do partial
  reductions either. Hence, this CombineFn has to do the work of batching,
  partial reduction (intro metrics), and actual combining, all in one.

  We do this by accumulating FeaturesPredictionsLabels in the combine state
  until we accumulate a large enough batch, at which point we send them
  through the "intro metrics" step. When merging, we merge the metric variables
  and accumulate FeaturesPredictionsLabels accordingly. We do one final
  "intro metrics" and merge step before producing the final output value.

  See also:
  BEAM-3737: Key-aware batching function.
  """

  # This needs to be large enough to allow for efficient TF invocations during
  # batch flushing, but shouldn't be too large as it could lead to large amout
  # of data being shuffled for non-flushed batches. Its value might be
  # increasable in the future though, after BEAM-4030 is resolved and
  # CombineFn.compact is appropriately implemented herein and the various
  # Beam Runners can make use of it.
  _DEFAULT_DESIRED_BATCH_SIZE = 100

  def __init__(self,
               eval_shared_model,
               desired_batch_size = None,
               compute_with_sampling = False):
    self._eval_shared_model = eval_shared_model
    self._eval_metrics_graph = None  # type: eval_metrics_graph.EvalMetricsGraph
    self._model_load_seconds = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')
    self._combine_batch_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'combine_batch_size')
    self._compute_with_sampling = compute_with_sampling

    # We really want the batch size to be adaptive like it is in
    # beam.BatchElements(), but there isn't an easy way to make it so.
    if desired_batch_size and desired_batch_size > 0:
      self._desired_batch_size = desired_batch_size
    else:
      self._desired_batch_size = self._DEFAULT_DESIRED_BATCH_SIZE

  def _start_bundle(self):
    # There's no initialisation method for CombineFns.
    # See BEAM-3736: Add SetUp() and TearDown() for CombineFns.
    self._eval_metrics_graph = (
        self._eval_shared_model.shared_handle.acquire(
            self._eval_shared_model.construct_fn))

  def _poissonify(self, accumulator):
    # pylint: disable=line-too-long
    """Creates a bootstrap resample of the data in an accumulator.

    Given a set of data, it will be represented in the resample set a number of
    times, that number of times is drawn from Poisson(1).
    See
    http://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html
    for a detailed explanation of the technique.

    Args:
      accumulator: Accumulator containing FPLs from a sample

    Returns:
      A list of FPLs representing a bootstrap resample of the accumulator items.
    """
    fpls_for_metrics = []
    poisson_values = np.random.poisson(1, len(accumulator.fpls))
    for i, fpl in enumerate(accumulator.fpls):
      for _ in range(0, poisson_values[i]):
        fpls_for_metrics.append(copy.copy(fpl))
    return fpls_for_metrics

  def _maybe_do_batch(self, accumulator,
                      force = False):
    """Maybe intro metrics and update accumulator in place.

    Checks if accumulator has enough FPLs for a batch, and if so, does the
    intro metrics for the batch and updates accumulator in place.

    Args:
      accumulator: Accumulator. Will be updated in place.
      force: Force intro metrics even if accumulator has less FPLs than the
        batch size.
    """

    if self._eval_metrics_graph is None:
      self._start_bundle()
    batch_size = len(accumulator.fpls)
    if force or batch_size >= self._desired_batch_size:
      if accumulator.fpls:
        self._combine_batch_size.update(batch_size)
        fpls_for_metrics = accumulator.fpls
        if self._compute_with_sampling:
          # If we are computing with multiple bootstrap replicates, use fpls
          # generated by the Poisson bootstrapping technique.
          fpls_for_metrics = self._poissonify(accumulator)
        if fpls_for_metrics:
          accumulator.add_metrics_variables(
              self._eval_metrics_graph.metrics_reset_update_get_list(
                  fpls_for_metrics))
          del accumulator.fpls[:]

  def create_accumulator(self):
    return _AggState()

  def add_input(self, accumulator,
                elem):
    accumulator.add_fpl(elem)
    self._maybe_do_batch(accumulator)
    return accumulator

  def merge_accumulators(self, accumulators):
    result = _AggState()
    for acc in accumulators:
      result += acc
      # Compact within the loop to avoid accumulating too much data.
      #
      # During the "map" side of combining combining happens per bundle,
      # but on the "reduce" side it's across all bundles (for a given key).
      #
      # So we could potentially accumulate get num_bundles * batch_size
      # elements if we don't process the batches within the loop, which
      # could cause OOM errors (b/77293756).
      self._maybe_do_batch(result)

    # Ensure (via 'force=True') that all "merged" accumulators that are produced
    # are compact and "fully" merged (ie ready for use for extract_output). Any
    # overhead that might be induced by this due to possibly small batches is
    # likely dwarfed by the overhead that materializing large accumulators
    # induces.
    self._maybe_do_batch(result, force=True)

    return result

  def extract_output(
      self, accumulator):

    # It's possible that the accumulator has not been fully flushed, if it was
    # not produced by a call to merge_accumulators (which is not guaranteed
    # across all Beam Runners), so we defensively flush it here again, before we
    # extract data from it, to ensure correctness.
    self._maybe_do_batch(accumulator, force=True)
    return accumulator.metric_variables


@beam.typehints.with_input_types(
    beam.typehints.Tuple[_BeamSliceKeyType, beam.typehints.List[beam.typehints
                                                                .Any]])
# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
class _ExtractOutputDoFn(beam.DoFn):
  """A DoFn that extracts the metrics output."""

  OUTPUT_TAG_METRICS = 'tag_metrics'
  OUTPUT_TAG_PLOTS = 'tag_plots'

  def __init__(self, eval_shared_model):
    self._eval_shared_model = eval_shared_model

  def start_bundle(self):
    # There's no initialisation method for CombineFns.
    # See BEAM-3736: Add SetUp() and TearDown() for CombineFns.
    self._eval_saved_model = (
        self._eval_shared_model.shared_handle.acquire(
            self._eval_shared_model.construct_fn))

  def process(
      self, element
  ):
    (slice_key, metric_variables) = element
    if metric_variables:
      self._eval_saved_model.set_metric_variables(metric_variables)
    # Note that there is a negligible chance of a poisson-bootstrap sample
    # being empty after multiple calls to _maybe_do_batch. In this case it will
    # return results from the last-computed slice. At reasonable sample sizes,
    # the odds of this are so astronomical it's barely worth considering much
    # less guarding against, but I'm putting this comment here to acknowledge
    # the possibility.
    result = self._eval_saved_model.get_metric_values()
    slicing_metrics = {}
    plots = {}
    for k, v in result.items():
      plot = False
      # We need to check for substrings here because metrics may have prefixes
      # based on multiple labels and/or heads.
      for subkey in metric_keys.PLOT_KEYS:
        if k.endswith(subkey):
          plots[k] = v
          plot = True
      if not plot:
        slicing_metrics[k] = v
    yield (slice_key, slicing_metrics)
    if plots:
      yield beam.pvalue.TaggedOutput(self.OUTPUT_TAG_PLOTS, (slice_key, plots))  # pytype: disable=bad-return-type
