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
"""API for Tensorflow Model Analysis."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import pickle
import tempfile



import apache_beam as beam
import six
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import version as tfma_version
from tensorflow_model_analysis.eval_saved_model import dofn
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
import tensorflow_model_analysis.post_export_metrics.metric_keys as metric_keys
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.writers import writer
from tensorflow_model_analysis.types_compat import Any, Dict, List, NamedTuple, Optional, Text, Tuple

from google.protobuf import json_format

# File names for files written out to the result directory.
_METRICS_OUTPUT_FILE = 'metrics'
_PLOTS_OUTPUT_FILE = 'plots'
_EVAL_CONFIG_FILE = 'eval_config'

# Keys for the serialized final dictionary.
_VERSION_KEY = 'tfma_version'
_EVAL_CONFIG_KEY = 'eval_config'


def _assert_tensorflow_version():
  """Check that we're using a compatible TF version."""
  # Fail with a clear error in case we are not using a compatible TF version.
  major, minor, _ = tf.__version__.split('.')
  major = int(major)
  minor = int(minor)
  okay = True
  if major != 1:
    okay = False
  if minor < 11:
    okay = False
  if not okay:
    raise RuntimeError(
        'Tensorflow version >= 1.11, < 2 is required. Found (%s). Please '
        'install the latest 1.x version from '
        'https://github.com/tensorflow/tensorflow. ' % tf.__version__)


EvalConfig = NamedTuple(  # pylint: disable=invalid-name
    'EvalConfig',
    [
        ('model_location',
         Text),  # The location of the model used for this evaluation
        ('data_location',
         Text),  # The location of the data used for this evaluation
        ('slice_spec', Optional[List[slicer.SingleSliceSpec]]
        ),  # The corresponding slice spec
        ('example_weight_metric_key',
         Text),  # The name of the metric that contains example weight
    ])


def _check_version(raw_final_dict, path):
  version = raw_final_dict.get(_VERSION_KEY)
  if version is None:
    raise ValueError(
        'could not find TFMA version in raw deserialized dictionary for '
        'file at %s' % path)
  # We don't actually do any checking for now, since we don't have any
  # compatibility issues.


def _serialize_eval_config(eval_config):
  final_dict = {}
  final_dict[_VERSION_KEY] = tfma_version.VERSION_STRING
  final_dict[_EVAL_CONFIG_KEY] = eval_config
  return pickle.dumps(final_dict)


def load_eval_config(output_path):
  serialized_record = six.next(
      tf.python_io.tf_record_iterator(
          os.path.join(output_path, _EVAL_CONFIG_FILE)))
  final_dict = pickle.loads(serialized_record)
  _check_version(final_dict, output_path)
  return final_dict[_EVAL_CONFIG_KEY]


EvalResult = NamedTuple(  # pylint: disable=invalid-name
    'EvalResult',
    [('slicing_metrics', List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]]),
     ('plots', List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]]),
     ('config', EvalConfig)])


class EvalResults(object):
  """Class for results from multiple model analysis run."""

  def __init__(self,
               results,
               mode = constants.UNKNOWN_EVAL_MODE):
    supported_modes = [
        constants.DATA_CENTRIC_MODE,
        constants.MODEL_CENTRIC_MODE,
    ]
    if mode not in supported_modes:
      raise ValueError('Mode ' + mode + ' must be one of ' +
                       Text(supported_modes))

    self._results = results
    self._mode = mode

  def get_results(self):
    return self._results

  def get_mode(self):
    return self._mode


def make_eval_results(results, mode):
  """Run model analysis for a single model on multiple data sets.

  Args:
    results: A list of TFMA evaluation results.
    mode: The mode of the evaluation. Currently, tfma.DATA_CENTRIC_MODE and
      tfma.MODEL_CENTRIC_MODE are supported.

  Returns:
    An EvalResults containing all evaluation results. This can be used to
    construct a time series view.
  """
  return EvalResults(results, mode)


def load_eval_results(output_paths, mode):
  """Run model analysis for a single model on multiple data sets.

  Args:
    output_paths: A list of output paths of completed tfma runs.
    mode: The mode of the evaluation. Currently, tfma.DATA_CENTRIC_MODE and
      tfma.MODEL_CENTRIC_MODE are supported.

  Returns:
    An EvalResults containing the evaluation results serialized at output_paths.
    This can be used to construct a time series view.
  """
  results = [load_eval_result(output_path) for output_path in output_paths]
  return make_eval_results(results, mode)


def load_eval_result(output_path):
  """Creates an EvalResult object for use with the visualization functions."""
  metrics_proto_list = metrics_and_plots_evaluator.load_and_deserialize_metrics(
      path=os.path.join(output_path, _METRICS_OUTPUT_FILE))
  plots_proto_list = metrics_and_plots_evaluator.load_and_deserialize_plots(
      path=os.path.join(output_path, _PLOTS_OUTPUT_FILE))

  slicing_metrics = [(key, _convert_metric_map_to_dict(metrics_data))
                     for key, metrics_data in metrics_proto_list]
  plots = [(key, json_format.MessageToDict(plot_data))
           for key, plot_data in plots_proto_list]

  eval_config = load_eval_config(output_path)
  return EvalResult(
      slicing_metrics=slicing_metrics, plots=plots, config=eval_config)


def default_eval_shared_model(
    eval_saved_model_path,
    add_metrics_callbacks = None,
    include_default_metrics = True,
    example_weight_key = None):
  """Returns default EvalSharedModel.

  Args:
    eval_saved_model_path: Path to EvalSavedModel.
    add_metrics_callbacks: Optional list of callbacks for adding additional
      metrics to the graph (see EvalSharedModel for more information on how to
      configure additional metrics). Metrics for example counts and example
      weight will be added automatically.
    include_default_metrics: True to include the default metrics that are part
      of the saved model graph during evaluation.
    example_weight_key: The key of the example weight column. If None, weight
      will be 1 for each example.
  """
  # Always compute example weight and example count.
  # pytype: disable=module-attr
  if not add_metrics_callbacks:
    add_metrics_callbacks = []
  example_count_callback = post_export_metrics.example_count()
  add_metrics_callbacks.append(example_count_callback)
  if example_weight_key:
    example_weight_callback = post_export_metrics.example_weight(
        example_weight_key)
    add_metrics_callbacks.append(example_weight_callback)
  # pytype: enable=module-attr

  return types.EvalSharedModel(
      model_path=eval_saved_model_path,
      add_metrics_callbacks=add_metrics_callbacks,
      include_default_metrics=include_default_metrics,
      example_weight_key=example_weight_key,
      construct_fn=dofn.make_construct_fn(eval_saved_model_path,
                                          add_metrics_callbacks,
                                          include_default_metrics))


def default_extractors(  # pylint: disable=invalid-name
    eval_shared_model,
    slice_spec = None,
    desired_batch_size = None,
    materialize = True):
  """Returns the default extractors for use in ExtractAndEvaluate.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel.
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    desired_batch_size: Optional batch size for batching in Aggregate.
    materialize: True to have extractors create materialized output.
  """
  return [
      predict_extractor.PredictExtractor(
          eval_shared_model, desired_batch_size, materialize=materialize),
      slice_key_extractor.SliceKeyExtractor(
          slice_spec, materialize=materialize)
  ]


def default_evaluators(  # pylint: disable=invalid-name
    eval_shared_model,
    desired_batch_size = None):
  """Returns the default evaluators for use in ExtractAndEvaluate.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional batch size for batching in Aggregate.
  """
  return [
      metrics_and_plots_evaluator.MetricsAndPlotsEvaluator(
          eval_shared_model, desired_batch_size)
  ]


def default_writers(output_path):  # pylint: disable=invalid-name
  """Returns the default writers for use in WriteResults.

  Args:
    output_path: Path to store results files under.
  """
  writers = []
  output = {
      constants.METRICS_KEY: os.path.join(output_path, _METRICS_OUTPUT_FILE),
      constants.PLOTS_KEY: os.path.join(output_path, _PLOTS_OUTPUT_FILE)
  }
  for (key, output_file) in output.items():
    writers.append(
        writer.Writer(
            stage_name='WriteTFRecord(%s)' % output_file,
            ptransform=writer.Write(
                key=key,
                ptransform=beam.io.WriteToTFRecord(
                    file_path_prefix=output_file, shard_name_template=''))))
  return writers


# The input type is a MessageMap where the keys are strings and the values are
# some protocol buffer field. Note that MessageMap is not a protobuf message,
# none of the exising utility methods work on it. We must iterate over its
# values and call the utility function individually.
def _convert_metric_map_to_dict(metric_map):
  """Converts a metric map (metrics in MetricsForSlice protobuf) into a dict.

  Args:
    metric_map: A protocol buffer MessageMap that has behaviors like dict. The
      keys are strings while the values are protocol buffers. However, it is not
      a protobuf message and cannot be passed into json_format.MessageToDict
      directly. Instead, we must iterate over its values.

  Returns:
    A dict representing the metric_map. For example:
    Assume myProto contains
    {
      metrics: {
        key: 'double'
        value: {
          double_value: {
            value: 1.0
          }
        }
      }
      metrics: {
        key: 'bounded'
        value: {
          bounded_value: {
            lower_bound: {
              double_value: {
                value: 0.8
              }
            }
            upper_bound: {
              double_value: {
                value: 0.9
              }
            }
            value: {
              double_value: {
                value: 0.86
              }
            }
          }
        }
      }
    }

    The output of _convert_metric_map_to_dict(myProto.metrics) would be

    {
      'double': {
        'doubleValue': 1.0,
      },
      'bounded': {
        'boundedValue': {
          'lowerBound': 0.8,
          'upperBound': 0.9,
          'value': 0.86,
        },
      },
    }

    Note that field names are converted to lowerCamelCase and the field value in
    google.protobuf.DoubleValue is collapsed automatically.
  """
  return {k: json_format.MessageToDict(metric_map[k]) for k in metric_map}


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(beam.typehints.Any)
def InputsToExtracts(  # pylint: disable=invalid-name
    inputs):
  """Converts serialized inputs (e.g. examples) to Extracts."""
  return inputs | beam.Map(lambda x: {constants.INPUT_KEY: x})


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(evaluator.Evaluation)
def ExtractAndEvaluate(  # pylint: disable=invalid-name
    extracts, extractors,
    evaluators):
  """Performs Extractions and Evaluations in provided order."""
  evaluation = {}
  # Run evaluators that run before extraction (i.e. that only require
  # the incoming input extract added by ReadInputs)
  for v in evaluators:
    if not v.run_after:
      evaluation.update(extracts | v.stage_name >> v.ptransform)
  for x in extractors:
    extracts = (extracts | x.stage_name >> x.ptransform)
    for v in evaluators:
      if v.run_after == x.stage_name:
        evaluation.update(extracts | v.stage_name >> v.ptransform)
  for v in evaluators:
    if v.run_after == constants.LAST_EXTRACTOR:
      evaluation.update(extracts | v.stage_name >> v.ptransform)
  return evaluation


@beam.ptransform_fn
@beam.typehints.with_input_types(evaluator.Evaluation)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def WriteResults(  # pylint: disable=invalid-name
    evaluation, writers):
  """Writes Evaluation results using given writers.

  Args:
    evaluation: Evaluation output.
    writers: Writes to use for writing out Evaluation output.

  Raises:
    ValueError: If Evaluation is empty.

  Returns:
    beam.pvalue.PDone.
  """
  if not evaluation:
    raise ValueError('Evaluation cannot be empty')
  for w in writers:
    _ = evaluation | w.stage_name >> w.ptransform
  return beam.pvalue.PDone(list(evaluation.values())[0].pipeline)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def WriteEvalConfig(  # pylint: disable=invalid-name
    pipeline, eval_config, output_path):
  """Writes EvalConfig to file.

  Args:
    pipeline: Beam pipeline.
    eval_config: EvalConfig.
    output_path: Path to store output under.

  Returns:
    beam.pvalue.PDone.
  """
  return (
      pipeline
      | 'CreateEvalConfig' >> beam.Create([_serialize_eval_config(eval_config)])
      | 'WriteEvalConfig' >> beam.io.WriteToTFRecord(
          os.path.join(output_path, _EVAL_CONFIG_FILE), shard_name_template=''))


@beam.ptransform_fn
@beam.typehints.with_output_types(beam.pvalue.PDone)
def ExtractEvaluateAndWriteResults(  # pylint: disable=invalid-name
    examples,
    eval_shared_model,
    output_path,
    display_only_data_location = None,
    slice_spec = None,
    desired_batch_size = None,
    extractors = None,
    evaluators = None,
    writers = None,
    write_config = True):
  """PTransform for performing extraction, evaluation, and writing results.

  Users who want to construct their own Beam pipelines instead of using the
  lightweight run_model_analysis functions should use this PTransform.

  Example usage:
    eval_shared_model = tfma.default_eval_shared_model(
        eval_saved_model_path=model_location,
        add_metrics_callbacks=[...],
        example_weight_key=example_weight_key)
    with beam.Pipeline(runner=...) as p:
      _ = (p
           | 'ReadData' >> beam.io.ReadFromTFRecord(data_location)
           | 'ExtractEvaluateAndWriteResults' >>
           tfma.ExtractEvaluateAndWriteResults(
               eval_shared_model=eval_shared_model,
               output_path=output_path,
               display_only_data_location=data_location,
               slice_spec=slice_spec,
               ...))
    result = tfma.load_eval_result(output_path=output_path)
    tfma.view.render_slicing_metrics(result)

  Note that the exact serialization format is an internal implementation detail
  and subject to change. Users should only use the TFMA functions to write and
  read the results.

  Args:
    examples: PCollection of input examples. Can be any format the model accepts
      (e.g. string containing CSV row, TensorFlow.Example, etc).
    eval_shared_model: Shared model parameters for EvalSavedModel including any
      additional metrics (see EvalSharedModel for more information on how to
      configure additional metrics).
    output_path: Path to output metrics and plots results.
    display_only_data_location: Optional path indicating where the examples were
      read from. This is used only for display purposes - data will not actually
      be read from this path.
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    desired_batch_size: Optional batch size for batching in Predict and
      Aggregate.
    extractors: Optional list of Extractors to apply to Extracts. Typically
      these will be added by calling the default_extractors function. If no
      extractors are provided, default_extractors (non-materialized) will be
      used.
    evaluators: Optional list of Evaluators for evaluating Extracts. Typically
      these will be added by calling the default_evaluators function. If no
      evaluators are provided, default_evaluators will be used.
    writers: Optional list of Writers for writing Evaluation output. Typically
      these will be added by calling the default_writers function. If no writers
      are provided, default_writers will be used.
    write_config: True to write the config along with the results.

  Raises:
    ValueError: If matching Extractor not found for an Evaluator.

  Returns:
    PDone.
  """
  if not extractors:
    extractors = default_extractors(
        eval_shared_model=eval_shared_model,
        slice_spec=slice_spec,
        desired_batch_size=desired_batch_size,
        materialize=False)

  if not evaluators:
    evaluators = default_evaluators(
        eval_shared_model=eval_shared_model,
        desired_batch_size=desired_batch_size)

  for v in evaluators:
    evaluator.verify_evaluator(v, extractors)

  if not writers:
    writers = default_writers(output_path=output_path)

  data_location = '<user provided PCollection>'
  if display_only_data_location is not None:
    data_location = display_only_data_location

  example_weight_metric_key = metric_keys.EXAMPLE_COUNT
  if eval_shared_model.example_weight_key:
    example_weight_metric_key = metric_keys.EXAMPLE_WEIGHT

  eval_config = EvalConfig(
      model_location=eval_shared_model.model_path,
      data_location=data_location,
      slice_spec=slice_spec,
      example_weight_metric_key=example_weight_metric_key)

  # pylint: disable=no-value-for-parameter
  _ = (
      examples
      | 'InputsToExtracts' >> InputsToExtracts()
      | 'ExtractAndEvaluate' >> ExtractAndEvaluate(
          extractors=extractors, evaluators=evaluators)
      | 'WriteResults' >> WriteResults(writers=writers))

  if write_config:
    _ = examples.pipeline | WriteEvalConfig(eval_config, output_path)
  # pylint: enable=no-value-for-parameter

  return beam.pvalue.PDone(examples.pipeline)


def run_model_analysis(
    eval_shared_model,
    data_location,
    file_format = 'tfrecords',
    slice_spec = None,
    output_path = None,
    extractors = None,
    evaluators = None,
    writers = None,
    write_config = True,
    pipeline_options = None,
):
  """Runs TensorFlow model analysis.

  It runs a Beam pipeline to compute the slicing metrics exported in TensorFlow
  Eval SavedModel and returns the results.

  This is a simplified API for users who want to quickly get something running
  locally. Users who wish to create their own Beam pipelines can use the
  Evaluate PTransform instead.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel including any
      additional metrics (see EvalSharedModel for more information on how to
      configure additional metrics).
    data_location: The location of the data files.
    file_format: The file format of the data, can be either 'text' or
      'tfrecords' for now. By default, 'tfrecords' will be used.
    slice_spec: A list of tfma.slicer.SingleSliceSpec. Each spec represents a
      way to slice the data. If None, defaults to the overall slice.
      Example usages:
      - tfma.SingleSiceSpec(): no slice, metrics are computed on overall data.
      - tfma.SingleSiceSpec(columns=['country']): slice based on features in
        column "country". We might get metrics for slice "country:us",
        "country:jp", and etc in results.
      - tfma.SingleSiceSpec(features=[('country', 'us')]): metrics are computed
        on slice "country:us".
    output_path: The directory to output metrics and results to. If None, we use
      a temporary directory.
    extractors: Optional list of Extractors to apply to Extracts. Typically
      these will be added by calling the default_extractors function. If no
      extractors are provided, default_extractors (non-materialized) will be
      used.
    evaluators: Optional list of Evaluators for evaluating Extracts. Typically
      these will be added by calling the default_evaluators function. If no
      evaluators are provided, default_evaluators will be used.
    writers: Optional list of Writers for writing Evaluation output. Typically
      these will be added by calling the default_writers function. If no writers
      are provided, default_writers will be used.
    write_config: True to write the config along with the results.
    pipeline_options: Optional arguments to run the Pipeline, for instance
      whether to run directly.

  Returns:
    An EvalResult that can be used with the TFMA visualization functions.

  Raises:
    ValueError: If the file_format is unknown to us.
  """
  _assert_tensorflow_version()
  # Get working_dir ready.
  if output_path is None:
    output_path = tempfile.mkdtemp()
  if not tf.gfile.Exists(output_path):
    tf.gfile.MakeDirs(output_path)

  with beam.Pipeline(options=pipeline_options) as p:
    if file_format == 'tfrecords':
      data = p | 'ReadFromTFRecord' >> beam.io.ReadFromTFRecord(
          file_pattern=data_location,
          compression_type=beam.io.filesystem.CompressionTypes.UNCOMPRESSED)
    elif file_format == 'text':
      data = p | 'ReadFromText' >> beam.io.textio.ReadFromText(data_location)
    else:
      raise ValueError('unknown file_format: %s' % file_format)

    # pylint: disable=no-value-for-parameter
    _ = (
        data
        | 'ExtractEvaluateAndWriteResults' >> ExtractEvaluateAndWriteResults(
            eval_shared_model=eval_shared_model,
            output_path=output_path,
            display_only_data_location=data_location,
            slice_spec=slice_spec,
            extractors=extractors,
            evaluators=evaluators,
            writers=writers,
            write_config=write_config))
    # pylint: enable=no-value-for-parameter

  eval_result = load_eval_result(output_path=output_path)
  return eval_result


def multiple_model_analysis(model_locations, data_location,
                            **kwargs):
  """Run model analysis for multiple models on the same data set.

  Args:
    model_locations: A list of paths to the export eval saved model.
    data_location: The location of the data files.
    **kwargs: The args used for evaluation. See tfma.run_model_analysis() for
      details.

  Returns:
    A tfma.EvalResults containing all the evaluation results with the same order
    as model_locations.
  """
  results = []
  for m in model_locations:
    results.append(
        run_model_analysis(
            default_eval_shared_model(m), data_location, **kwargs))
  return EvalResults(results, constants.MODEL_CENTRIC_MODE)


def multiple_data_analysis(model_location, data_locations,
                           **kwargs):
  """Run model analysis for a single model on multiple data sets.

  Args:
    model_location: The location of the exported eval saved model.
    data_locations: A list of data set locations.
    **kwargs: The args used for evaluation. See tfma.run_model_analysis() for
      details.

  Returns:
    A tfma.EvalResults containing all the evaluation results with the same order
    as data_locations.
  """
  results = []
  for d in data_locations:
    results.append(
        run_model_analysis(
            default_eval_shared_model(model_location), d, **kwargs))
  return EvalResults(results, constants.DATA_CENTRIC_MODE)
