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
"""Serialization library. For internal use only."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
import numpy as np
import six
import tensorflow as tf

from tensorflow_model_analysis import types
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import Any, Dict, List, Text, Tuple


def deserialize_slice_key(
    slice_key):
  """Converts proto SliceKey to slicer.SliceKeyType.

  Args:
    slice_key: The slice key in the format of proto SliceKey.

  Returns:
    The slice key in the format of slicer.SliceKeyType.

  Raises:
    TypeError: If the evaluate type is unreconized.
  """
  result = []
  for elem in slice_key.single_slice_keys:
    if elem.HasField('bytes_value'):
      value = elem.bytes_value
    elif elem.HasField('int64_value'):
      value = elem.int64_value
    elif elem.HasField('float_value'):
      value = elem.float_value
    else:
      raise TypeError(
          'unrecognized type of type %s, value %s' % (type(elem), elem))
    result.append((tf.compat.as_bytes(elem.column), value))
  return tuple(result)


def load_and_deserialize_metrics(
    path):
  result = []
  for record in tf.python_io.tf_record_iterator(path):
    metrics_for_slice = metrics_for_slice_pb2.MetricsForSlice.FromString(record)
    result.append((
        deserialize_slice_key(metrics_for_slice.slice_key),  # pytype: disable=wrong-arg-types
        metrics_for_slice.metrics))
  return result


def load_and_deserialize_plots(
    path):
  """Returns deserialized plots loaded from given path."""
  result = []
  for record in tf.python_io.tf_record_iterator(path):
    plots_for_slice = metrics_for_slice_pb2.PlotsForSlice.FromString(record)
    result.append((
        deserialize_slice_key(plots_for_slice.slice_key),  # pytype: disable=wrong-arg-types
        plots_for_slice.plot_data))
  return result


def _convert_slice_key(
    slice_key):
  """Converts slice_key into metrics_for_slice_pb2.SliceKey proto."""
  result = metrics_for_slice_pb2.SliceKey()

  for (col, val) in slice_key:
    single_slice_key = result.single_slice_keys.add()
    single_slice_key.column = col
    if isinstance(val, (six.binary_type, six.text_type)):
      single_slice_key.bytes_value = tf.compat.as_bytes(val)
    elif isinstance(val, six.integer_types):
      single_slice_key.int64_value = val
    elif isinstance(val, float):
      single_slice_key.float_value = val
    else:
      raise TypeError(
          'unrecognized type of type %s, value %s' % (type(val), val))

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
    if isinstance(value, (six.binary_type, six.text_type)):
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
  result.slice_key.CopyFrom(_convert_slice_key(slice_key))

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
  result.slice_key.CopyFrom(_convert_slice_key(slice_key))

  # Convert the slice plots.
  _convert_slice_plots(slice_plots, post_export_metrics, result.plot_data)  # pytype: disable=wrong-arg-types

  return result.SerializeToString()


# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
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
