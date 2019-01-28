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
"""Public API for performing metrics and plots evaluations."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import apache_beam as beam
import numpy as np
import six
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import aggregate
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import Any, Dict, List, Optional, Text, Tuple


def MetricsAndPlotsEvaluator(  # pylint: disable=invalid-name
    eval_shared_model,
    desired_batch_size = None,
    metrics_key = constants.METRICS_KEY,
    plots_key = constants.PLOTS_KEY,
    run_after = slice_key_extractor.SLICE_KEY_EXTRACTOR_STAGE_NAME
):
  """Creates an Evaluator for evaluating metrics and plots.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional batch size for batching in Aggregate.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.
    run_after: Extractor to run after (None means before any extractors).

  Returns:
    Evaluator for evaluating metrics and plots. The output will be stored under
    'metrics' and 'plots' keys.
  """
  # pylint: disable=no-value-for-parameter
  return evaluator.Evaluator(
      stage_name='EvaluateMetricsAndPlots',
      run_after=run_after,
      ptransform=EvaluateMetricsAndPlots(
          eval_shared_model=eval_shared_model,
          desired_batch_size=desired_batch_size,
          metrics_key=metrics_key,
          plots_key=plots_key))
  # pylint: enable=no-value-for-parameter


def load_and_deserialize_metrics(
    path):
  result = []
  for record in tf.python_io.tf_record_iterator(path):
    metrics_for_slice = metrics_for_slice_pb2.MetricsForSlice.FromString(record)
    result.append((
        slicer.deserialize_slice_key(metrics_for_slice.slice_key),  # pytype: disable=wrong-arg-types
        metrics_for_slice.metrics))
  return result


def load_and_deserialize_plots(
    path):
  """Returns deserialized plots loaded from given path."""
  result = []
  for record in tf.python_io.tf_record_iterator(path):
    plots_for_slice = metrics_for_slice_pb2.PlotsForSlice.FromString(record)
    result.append((
        slicer.deserialize_slice_key(plots_for_slice.slice_key),  # pytype: disable=wrong-arg-types
        plots_for_slice.plot_data))
  return result


def _convert_to_array_value(
    array):
  """Converts NumPy array to ArrayValue."""
  result = metrics_for_slice_pb2.ArrayValue()
  result.shape[:] = array.shape
  if array.dtype == 'int32':
    result.data_type = metrics_for_slice_pb2.ArrayValue.INT32
    result.int32_values[:] = array.flatten()
  elif array.dtype == 'int64':
    result.data_type = metrics_for_slice_pb2.ArrayValue.INT64
    result.int64_values[:] = array.flatten()
  elif array.dtype == 'float32':
    result.data_type = metrics_for_slice_pb2.ArrayValue.FLOAT32
    result.float32_values[:] = array.flatten()
  elif array.dtype == 'float64':
    result.data_type = metrics_for_slice_pb2.ArrayValue.FLOAT64
    result.float64_values[:] = array.flatten()
  else:
    # For all other types, cast to string and convert to bytes.
    result.data_type = metrics_for_slice_pb2.ArrayValue.BYTES
    result.bytes_values[:] = [
        tf.compat.as_bytes(x) for x in array.astype(six.text_type).flatten()
    ]
  return result


def _convert_slice_metrics(
    slice_metrics,
    post_export_metrics,
    metrics_for_slice):
  """Converts slice_metrics into the given metrics_for_slice proto."""
  # Convert the metrics from post_export_metrics to the structured output if
  # defined.
  for post_export_metric in post_export_metrics:
    if hasattr(post_export_metric, 'populate_stats_and_pop'):
      post_export_metric.populate_stats_and_pop(slice_metrics,
                                                metrics_for_slice.metrics)

  for name, value in slice_metrics.items():
    if isinstance(value, types.ValueWithConfidenceInterval):
      # Convert to a bounded value.
      metrics_for_slice.metrics[name].bounded_value.value.value = value.value
      metrics_for_slice.metrics[
          name].bounded_value.lower_bound.value = value.lower_bound
      metrics_for_slice.metrics[
          name].bounded_value.upper_bound.value = value.upper_bound
      metrics_for_slice.metrics[name].bounded_value.methodology = (
          metrics_for_slice_pb2.BoundedValue.POISSON_BOOTSTRAP)
    elif isinstance(value, (six.binary_type, six.text_type)):
      # Convert textual types to string metrics.
      metrics_for_slice.metrics[name].bytes_value = value
    elif isinstance(value, np.ndarray):
      # Convert NumPy arrays to ArrayValue.
      metrics_for_slice.metrics[name].array_value.CopyFrom(
          _convert_to_array_value(value))
    else:
      # We try to convert to float values.
      try:
        metrics_for_slice.metrics[name].double_value.value = float(value)
      except (TypeError, ValueError) as e:
        metrics_for_slice.metrics[name].unknown_type.value = str(value)
        metrics_for_slice.metrics[name].unknown_type.error = e.message


def _serialize_metrics(
    metrics,
    post_export_metrics):
  """Converts the given slice metrics into serialized proto MetricsForSlice.

  Args:
    metrics: The slice metrics.
    post_export_metrics: A list of metric callbacks. This should be the same
      list as the one passed to tfma.Evaluate().

  Returns:
    The serialized proto MetricsForSlice.

  Raises:
    TypeError: If the type of the feature value in slice key cannot be
      recognized.
  """
  result = metrics_for_slice_pb2.MetricsForSlice()
  slice_key, slice_metrics = metrics

  # Convert the slice key.
  result.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))

  # Convert the slice metrics.
  _convert_slice_metrics(slice_metrics, post_export_metrics, result)

  return result.SerializeToString()


def _convert_slice_plots(
    slice_plots,
    post_export_metrics,
    plot_data):
  """Converts slice_plots into the given plot_data proto."""
  for post_export_metric in post_export_metrics:
    if hasattr(post_export_metric, 'populate_plots_and_pop'):
      post_export_metric.populate_plots_and_pop(slice_plots, plot_data)

  if slice_plots:
    raise NotImplementedError(
        'some plots were not converted or popped. keys: %s. post_export_metrics'
        'were: %s' % (
            slice_plots.keys(),
            [
                x.name for x in post_export_metrics  # pytype: disable=attribute-error
            ]))


def _serialize_plots(
    plots,
    post_export_metrics):
  """Converts the given slice plots into serialized proto PlotsForSlice..

  Args:
    plots: The slice plots.
    post_export_metrics: A list of metric callbacks. This should be the same
      list as the one passed to tfma.Evaluate().

  Returns:
    The serialized proto PlotsForSlice.
  """
  result = metrics_for_slice_pb2.PlotsForSlice()
  slice_key, slice_plots = plots

  # Convert the slice key.
  result.slice_key.CopyFrom(slicer.serialize_slice_key(slice_key))

  # Convert the slice plots.
  _convert_slice_plots(slice_plots, post_export_metrics, result.plot_data)  # pytype: disable=wrong-arg-types

  return result.SerializeToString()


# No typehint for input type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
@beam.typehints.with_output_types(beam.typehints.Tuple[bytes, bytes])
class SerializeMetricsAndPlots(beam.PTransform):  # pylint: disable=invalid-name
  """Converts metrics and plots into serialized protos."""

  def __init__(self, post_export_metrics):
    self._post_export_metrics = post_export_metrics

  def expand(
      self,
      metrics_and_plots
  ):
    """Converts the given metrics_and_plots into serialized proto.

    Args:
      metrics_and_plots: A Tuple of (slice metrics, slice plots).

    Returns:
      A Tuple of PCollection of Serialized proto MetricsForSlice.
    """
    metrics, plots = metrics_and_plots
    metrics = metrics | 'SerializeMetrics' >> beam.Map(
        _serialize_metrics, post_export_metrics=self._post_export_metrics)
    plots = plots | 'SerializePlots' >> beam.Map(
        _serialize_plots, post_export_metrics=self._post_export_metrics)
    return (metrics, plots)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
def ComputeMetricsAndPlots(  # pylint: disable=invalid-name
    extracts,
    eval_shared_model,
    desired_batch_size = None,
    num_bootstrap_samples = 1,
    random_seed = None,
):
  """Computes metrics and plots using the EvalSavedModel.

  Args:
    extracts: PCollection of Extracts. The extracts MUST contain a
      FeaturesPredictionsLabels extract keyed by
      tfma.FEATURE_PREDICTIONS_LABELS_KEY and a list of SliceKeyType extracts
      keyed by tfma.SLICE_KEY_TYPES_KEY. Typically these will be added by
      calling the default_extractors function.
    eval_shared_model: Shared model parameters for EvalSavedModel including any
      additional metrics (see EvalSharedModel for more information on how to
      configure additional metrics).
    desired_batch_size: Optional batch size for batching in Aggregate.
    num_bootstrap_samples: Set to value > 1 to run metrics analysis over
      multiple bootstrap samples and compute uncertainty intervals.
    random_seed: Provide for deterministic tests only.

  Returns:
    DoOutputsTuple. The tuple entries are
    PCollection of (slice key, metrics) and
    PCollection of (slice key, plot metrics).
  """
  # pylint: disable=no-value-for-parameter
  return (
      extracts

      # Input: one example at a time, with slice keys in extracts.
      # Output: one fpl example per slice key (notice that the example turns
      #         into n, replicated once per applicable slice key)
      | 'FanoutSlices' >> slicer.FanoutSlices()

      # Each slice key lands on one shard where metrics are computed for all
      # examples in that shard -- the "map" and "reduce" parts of the
      # computation happen within this shard.
      # Output: Multi-outputs, a dict of slice key to computed metrics, and
      # plots if applicable.
      | 'ComputePerSliceMetrics' >> aggregate.ComputePerSliceMetrics(
          eval_shared_model=eval_shared_model,
          desired_batch_size=desired_batch_size,
          num_bootstrap_samples=num_bootstrap_samples,
          random_seed=random_seed))
  # pylint: enable=no-value-for-parameter


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(evaluator.Evaluation)
def EvaluateMetricsAndPlots(  # pylint: disable=invalid-name
    extracts,
    eval_shared_model,
    desired_batch_size = None,
    metrics_key = constants.METRICS_KEY,
    plots_key = constants.PLOTS_KEY):
  """Evaluates metrics and plots using the EvalSavedModel.

  Args:
    extracts: PCollection of Extracts. The extracts MUST contain a
      FeaturesPredictionsLabels extract keyed by
      tfma.FEATURE_PREDICTION_LABELS_KEY and a list of SliceKeyType extracts
      keyed by tfma.SLICE_KEY_TYPES_KEY. Typically these will be added by
      calling the default_extractors function.
    eval_shared_model: Shared model parameters for EvalSavedModel including any
      additional metrics (see EvalSharedModel for more information on how to
      configure additional metrics).
    desired_batch_size: Optional batch size for batching in Aggregate.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.

  Returns:
    Evaluation containing serialized protos keyed by 'metrics' and 'plots'.
  """

  # pylint: disable=no-value-for-parameter
  metrics, plots = (
      extracts
      | 'Filter' >> extractor.Filter(include=[
          constants.FEATURES_PREDICTIONS_LABELS_KEY,
          constants.SLICE_KEY_TYPES_KEY
      ])
      | 'ComputeMetricsAndPlots' >> ComputeMetricsAndPlots(
          eval_shared_model, desired_batch_size))
  metrics, plots = (
      (metrics, plots)
      | 'SerializeMetricsAndPlots' >> SerializeMetricsAndPlots(
          post_export_metrics=eval_shared_model.add_metrics_callbacks))
  # pylint: enable=no-value-for-parameter

  return {metrics_key: metrics, plots_key: plots}
