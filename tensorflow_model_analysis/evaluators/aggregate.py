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
from scipy import mean
from scipy.stats import sem
from scipy.stats import t

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
from tensorflow_model_analysis.eval_saved_model import dofn
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import Any, Dict, Generator, Iterable, List, Optional, Text, Tuple, Union


_SAMPLE_ID = '___SAMPLE_ID'


@beam.ptransform_fn
@beam.typehints.with_input_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, slicer.BeamExtractsType])
@beam.typehints.with_output_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, beam.typehints
                         .List[beam.typehints.Any]])
def ComputePerSliceMetrics(  # pylint: disable=invalid-name
    slice_result,
    eval_shared_model,
    desired_batch_size = None,
    num_bootstrap_samples = 1,
    random_seed = None,
):
  """PTransform for computing, aggregating and combining metrics.

  Args:
    slice_result: Incoming PCollection consisting of slice key and extracts.
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional batch size for batching in Aggregate.
    num_bootstrap_samples: Number of replicas to use in calculating uncertainty
      using bootstrapping.  If 1 is provided (default), aggregate metrics will
      be calculated with no uncertainty. If num_bootstrap_samples is > 0,
      multiple samples of each slice will be calculated using the Poisson
      bootstrap method. To calculate standard errors, num_bootstrap_samples
      should be 20 or more in order to provide useful data. More is better, but
      you pay a performance cost.
    random_seed: Seed to use for testing, because nondeterministic tests stink.

  Returns:
    DoOutputsTuple. The tuple entries are
    PCollection of (slice key, metrics) and
    PCollection of (slice key, plot metrics).
  """
  slice_result.element_type = beam.typehints.Any

  compute_with_sampling = False
  if not num_bootstrap_samples:
    num_bootstrap_samples = 1
  if num_bootstrap_samples < 1:
    raise ValueError(
        'num_bootstrap_samples should be > 0, got %d' % num_bootstrap_samples)

  if num_bootstrap_samples > 1:
    slice_result_sampled = slice_result | 'FanoutBootstrap' >> beam.ParDo(
        _FanoutBootstrapFn(num_bootstrap_samples))
    compute_with_sampling = True

  fanout = 16

  output_results = (
      slice_result
      | 'CombinePerSlice' >> beam.CombinePerKey(
          _AggregateCombineFn(
              eval_shared_model=eval_shared_model,
              desired_batch_size=desired_batch_size,
              compute_with_sampling=False)).with_hot_key_fanout(fanout)
      | 'InterpretOutput' >> beam.ParDo(
          _ExtractOutputDoFn(eval_shared_model=eval_shared_model)))
  if compute_with_sampling:
    output_results = (
        slice_result_sampled
        | 'CombinePerSliceWithSamples' >> beam.CombinePerKey(
            _AggregateCombineFn(
                eval_shared_model=eval_shared_model,
                desired_batch_size=desired_batch_size,
                compute_with_sampling=True,
                seed_for_testing=random_seed)).with_hot_key_fanout(fanout)
        | 'InterpretSampledOutput' >> beam.ParDo(
            _ExtractOutputDoFn(eval_shared_model=eval_shared_model))
        | beam.GroupByKey()
        | beam.ParDo(_MergeBootstrap(), beam.pvalue.AsIter(output_results)))
  # Separate metrics and plots.
  return (output_results
          | beam.ParDo(_SeparateMetricsAndPlotsFn()).with_outputs(
              _SeparateMetricsAndPlotsFn.OUTPUT_TAG_PLOTS,
              main=_SeparateMetricsAndPlotsFn.OUTPUT_TAG_METRICS))


@beam.typehints.with_input_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, slicer.BeamExtractsType])
@beam.typehints.with_output_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, slicer.BeamExtractsType])
class _FanoutBootstrapFn(beam.DoFn):
  """For each bootstrap sample you want, we fan out an additional slice."""

  def __init__(self, num_bootstrap_samples):
    self._num_bootstrap_samples = num_bootstrap_samples

  def process(
      self, element
  ):
    slice_key, value = element
    for i in range(0, self._num_bootstrap_samples):
      # Prepend the sample ID to the original slice key.
      augmented_slice_key = ((_SAMPLE_ID, i), slice_key)
      # This fans out the pipeline, but because we are reducing in a
      # CombinePerKey, we shouldn't have to deal with a great increase in
      # network traffic.
      yield (augmented_slice_key, value)


def _collect_metrics(item, cumulative_key,
                     aggregated_metrics):
  """Aggregates individual metrics over multiple bootstrap samples.

  Since some metric values are compound, we have to make sure to aggregate
  each individual value in a compound metric and get its upper and lower bounds.
  We do this by creating a key based on the location of the value inside the
  compound metric, and storing the sample values corresponding to that key.

  Args:
    item: The metric value to add to aggregated_metrics.
    cumulative_key: The key (so far) to use in aggregated_metrics dict.
    aggregated_metrics: The dict collecting all the metrics over samples.
  """
  if isinstance(item, np.ndarray):
    # The metric is compound, we need to recurse until we hit individual values.
    for index, sub_item in enumerate(item):
      _collect_metrics(sub_item, '%s,%d' % (cumulative_key, index),
                       aggregated_metrics)
  else:
    # The metric is an individual value, and should be added to the aggregate
    # collection.
    if cumulative_key not in aggregated_metrics:
      aggregated_metrics[cumulative_key] = []
    if not np.isnan(item):
      aggregated_metrics[cumulative_key].append(item)


def _populate_bounded_metrics(
    index_list, metric_structure,
    value):
  """Recreates the original metric structure with bounded values."""
  if not index_list:
    metric_structure = value
    return
  if len(index_list) == 1:
    metric_structure[int(index_list[0])] = value
    return
  metric_structure = metric_structure[int(index_list.pop(0))]
  _populate_bounded_metrics(index_list, metric_structure, value)


class _MergeBootstrap(beam.DoFn):
  """Merge the bootstrap values and fit a T-distribution to get confidence."""

  def process(self, element, unsampled_results):
    slice_key, metrics = element
    # metrics should be a list of dicts, but the dataflow runner has a quirk
    # that requires specific casting.
    metrics = list(metrics)
    side_input_results = {}

    for result in unsampled_results:
      unsampled_slice_key, unsampled_metrics = result
      side_input_results[unsampled_slice_key] = unsampled_metrics
    if len(metrics) == 1:
      yield slice_key, metrics
      return

    original_structure = copy.copy(metrics[0])
    uber_metrics = {}
    unsampled_metrics = {}
    for m_dict in metrics:
      # For each metric in each slice, aggregate values over all of the computed
      # samples.
      for key in m_dict:
        _collect_metrics(m_dict[key], key, uber_metrics)
        unsampled_slice_key = slice_key
        if slice_key not in side_input_results:
          # Due to inconsistencies in Tuple casting in slice_key extractors.
          unsampled_slice_key = slice_key[0]
        _collect_metrics(side_input_results[unsampled_slice_key][key], key,
                         unsampled_metrics)

    for key in uber_metrics:
      # Compute confidence interval given the data points per metric.
      confidence = 0.95
      data = uber_metrics[key]
      # Data has to be numeric. That means throw out nan values.
      n_samples = len(data)
      if n_samples:
        sample_mean = mean(data)
        std_err = sem(data)
        t_stat = t.ppf((1 + confidence) / 2, n_samples - 1)
        upper_bound = sample_mean + t_stat * std_err
        lower_bound = sample_mean - t_stat * std_err
        # Set [mean, lower_bound, upper_bound] for each metric component.
        uber_metrics[key] = types.ValueWithConfidenceInterval(
            sample_mean, lower_bound, upper_bound, unsampled_metrics[key][0])
      else:
        uber_metrics[key] = types.ValueWithConfidenceInterval(
            float('nan'), float('nan'), float('nan'), float('nan'))

    # Convert metrics back into expected format with bounded values.
    for sub_key in uber_metrics:
      # Break sub-key into components.
      key_components = sub_key.split(',')
      original_key = key_components[0]
      metric_structure = original_structure[original_key]
      if isinstance(metric_structure, np.ndarray):
        metric_structure = np.array(metric_structure, dtype=object)
        _populate_bounded_metrics(key_components[1:], metric_structure,
                                  uber_metrics[sub_key])
      else:
        metric_structure = uber_metrics[sub_key]
      original_structure[original_key] = metric_structure

    yield slice_key, original_structure


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
    self.fpls = []  # type: List[beam.typehints.Any]

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


@beam.typehints.with_input_types(slicer.BeamExtractsType)
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
  BEAM-3737: Key-aware batching function
  (https://issues.apache.org/jira/browse/BEAM-3737).
  """

  # This needs to be large enough to allow for efficient TF invocations during
  # batch flushing, but shouldn't be too large as it also acts as cap on the
  # maximum memory usage of the computation.
  _DEFAULT_DESIRED_BATCH_SIZE = 100

  def __init__(self,
               eval_shared_model,
               desired_batch_size = None,
               compute_with_sampling = False,
               seed_for_testing = None):
    self._eval_shared_model = eval_shared_model
    self._eval_metrics_graph = None  # type: eval_metrics_graph.EvalMetricsGraph
    self._seed_for_testing = seed_for_testing
    # We really want the batch size to be adaptive like it is in
    # beam.BatchElements(), but there isn't an easy way to make it so.
    if desired_batch_size and desired_batch_size > 0:
      self._desired_batch_size = desired_batch_size
    else:
      self._desired_batch_size = self._DEFAULT_DESIRED_BATCH_SIZE

    self._compute_with_sampling = compute_with_sampling

    # Metrics.
    self._combine_batch_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'combine_batch_size')
    self._model_load_seconds = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')
    self._num_compacts = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_compacts')
    # This keeps track of the number of times the poisson bootstrap retries
    # when an empty set of elements is encountered. Should be extremely rare in
    # practice, keeping this counter will help us understand if something is
    # misbehaving.
    self._num_bootstrap_retries = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_bootstrap_retries')

  def _start_bundle(self):
    # There's no initialisation method for CombineFns.
    # See BEAM-3736: Add SetUp() and TearDown() for CombineFns.
    # Default to eval_saved_model dofn to preserve legacy assumption
    # of eval_saved_model.
    if self._eval_shared_model.construct_fn is None:
      construct_fn = dofn.make_construct_fn(
          eval_saved_model_path=self._eval_shared_model.model_path,
          add_metrics_callbacks=self._eval_shared_model.add_metrics_callbacks,
          include_default_metrics=True)
      self._eval_metrics_graph = (
          self._eval_shared_model.shared_handle.acquire(
              construct_fn(self._model_load_seconds)))
    else:
      self._eval_metrics_graph = (
          self._eval_shared_model.shared_handle.acquire(
              self._eval_shared_model.construct_fn(self._model_load_seconds)))

  def _poissonify(
      self, accumulator):
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
    if self._seed_for_testing:
      np.random.seed(self._seed_for_testing)

    result = []
    if accumulator.fpls:
      while not result:
        # It's possible that the distribution will roll zeroes for all of the
        # examples. In this case we will retry until we have at least one
        # element chosen. This is to avoid empty metric updates which currently
        # aren't handled well and should not have an significant statistical
        # impact.
        poisson_counts = np.random.poisson(1, len(accumulator.fpls))
        for i, fpl in enumerate(accumulator.fpls):
          result.extend([fpl] * poisson_counts[i])
        if not result:
          self._num_bootstrap_retries.inc(1)
    return result

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
    accumulator.add_fpl(elem[constants.FEATURES_PREDICTIONS_LABELS_KEY])
    self._maybe_do_batch(accumulator)
    return accumulator

  def merge_accumulators(self, accumulators):
    result = self.create_accumulator()
    for acc in accumulators:
      result += acc
      # Compact within the loop to avoid accumulating too much data.
      #
      # During the "map" side of combining merging happens with memory limits
      # but on the "reduce" side it's across all bundles (for a given key).
      #
      # So we could potentially accumulate get num_bundles * batch_size
      # elements if we don't process the batches within the loop, which
      # could cause OOM errors (b/77293756).
      self._maybe_do_batch(result)
    return result

  def compact(self, accumulator):
    self._maybe_do_batch(accumulator, force=True)  # Guaranteed compaction.
    self._num_compacts.inc(1)
    return accumulator

  def extract_output(
      self, accumulator):
    # It's possible that the accumulator has not been fully flushed, if it was
    # not produced by a call to compact (which is not guaranteed across all Beam
    # Runners), so we defensively flush it here again, before we extract data
    # from it, to ensure correctness.
    self._maybe_do_batch(accumulator, force=True)
    return accumulator.metric_variables


class _SeparateMetricsAndPlotsFn(beam.DoFn):
  """Separates metrics and plots into two separate PCollections."""
  OUTPUT_TAG_METRICS = 'tag_metrics'
  OUTPUT_TAG_PLOTS = 'tag_plots'

  def process(self,
              element):
    (slice_key, results) = element
    slicing_metrics = {}
    plots = {}
    for k, v in results.items():  # pytype: disable=attribute-error
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


@beam.typehints.with_input_types(
    beam.typehints.Tuple[slicer.BeamSliceKeyType, beam.typehints
                         .List[beam.typehints.Any]])
# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
class _ExtractOutputDoFn(beam.DoFn):
  """A DoFn that extracts the metrics output."""

  def __init__(self, eval_shared_model):
    self._eval_shared_model = eval_shared_model
    self._model_load_seconds = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def start_bundle(self):
    # There's no initialisation method for CombineFns.
    # See BEAM-3736: Add SetUp() and TearDown() for CombineFns.
    # Default to eval_saved_model dofn to preserve legacy assumption
    # of eval_saved_model.
    if self._eval_shared_model.construct_fn is None:
      construct_fn = dofn.make_construct_fn(
          eval_saved_model_path=self._eval_shared_model.model_path,
          add_metrics_callbacks=self._eval_shared_model.add_metrics_callbacks,
          include_default_metrics=True)
      self._eval_saved_model = (
          self._eval_shared_model.shared_handle.acquire(
              construct_fn(self._model_load_seconds)))
    else:
      self._eval_saved_model = (
          self._eval_shared_model.shared_handle.acquire(
              self._eval_shared_model.construct_fn(self._model_load_seconds)))

  def process(
      self, element
  ):
    (slice_key, metric_variables) = element
    if metric_variables:
      self._eval_saved_model.set_metric_variables(metric_variables)
    result = self._eval_saved_model.get_metric_values()

    # If slice key contains uncertainty sample ID, remove it from the key.
    if len(slice_key) and _SAMPLE_ID in slice_key[0]:
      slice_key = slice_key[1:]
    if len(slice_key) and not slice_key[0]:  # Overall slice.
      slice_key = ()
    yield (tuple(slice_key), result)
