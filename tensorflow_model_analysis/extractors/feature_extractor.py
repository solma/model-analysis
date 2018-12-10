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
"""Implements API for extracting features from an example."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy


import apache_beam as beam
import numpy as np
import tensorflow as tf

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.types_compat import Any, Dict, List, Optional, Text

# For now, we store only the first N sparse keys in our diagnostics table.
_MAX_SPARSE_FEATURES_PER_COLUMN = 10


def FeatureExtractor(excludes = None):
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name='ExtractFeatures',
      ptransform=ExtractFeatures(excludes=excludes))
  # pylint: enable=no-value-for-parameter


def _AugmentExtracts(fpl_dict,
                     excludes,
                     extracts):
  """Augments the Extracts with FeaturesPredictionsLabels.

  Args:
    fpl_dict: The dictionary returned by evaluate._Predict().
    excludes: List of strings containing features, predictions, or labels to
      exclude from materialization.
    extracts: The Extracts to be augmented. This is mutated in-place.

  Raises:
    TypeError: If the FeaturesPredictionsLabels is corrupt.
  """
  for name, val in fpl_dict.items():
    if excludes is not None and name in excludes:
      continue
    val = val.get(encoding.NODE_SUFFIX)

    if isinstance(val, tf.SparseTensorValue):
      extracts[name] = types.MaterializedColumn(
          name=name, value=val.values[0:_MAX_SPARSE_FEATURES_PER_COLUMN])

    elif isinstance(val, np.ndarray):
      val = val[0]  # only support first dim for now.
      if not np.isscalar(val):
        val = val[0:_MAX_SPARSE_FEATURES_PER_COLUMN]
      extracts[name] = types.MaterializedColumn(name=name, value=val)

    else:
      raise TypeError(
          'Dictionary item with key %s, value %s had unexpected type %s' %
          (name, val, type(val)))


def _MaterializeFeatures(
    extracts,
    excludes = None):
  """Converts FeaturesPredictionsLabels into MaterializedColumn in the extract.

  It must be the case that evaluate._Predict() was called on the Extracts before
  calling this function.

  Args:
    extracts: The Extracts to be augmented.
    excludes: Optional list of strings containing features, predictions, or
      labels to exclude from materialization.

  Returns:
    Returns an augmented Extracts (which is a shallow copy of the original
    Extracts, so the original isn't mutated)

  Raises:
    RuntimeError: When _Predict() didn't populate the 'fpl' key.
  """
  # Make a a shallow copy, so we don't mutate the original.
  result = copy.copy(extracts)

  fpl = result.get(constants.FEATURES_PREDICTIONS_LABELS_KEY)
  if not fpl:
    raise RuntimeError('FPL missing, Please ensure _Predict() was called.')

  if not isinstance(fpl, types.FeaturesPredictionsLabels):
    raise TypeError(
        'Expected FPL to be instance of FeaturesPredictionsLabel. FPL was: %s '
        'of type %s' % (str(fpl), type(fpl)))

  # We disable pytyping here because we know that 'fpl' key corresponds to a
  # non-materialized column.
  # pytype: disable=attribute-error
  _AugmentExtracts(fpl.features, excludes, result)
  _AugmentExtracts(fpl.predictions, excludes, result)
  _AugmentExtracts(fpl.labels, excludes, result)
  return result
  # pytype: enable=attribute-error


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(beam.typehints.Any)
def ExtractFeatures(
    extracts,
    excludes = None):
  """Builds MaterializedColumn extracts from FPL created in evaluate.Predict().

  It must be the case that evaluate._Predict() was called on the Extracts before
  calling this function.

  Args:
    extracts: PCollection containing the Extracts that will have
      MaterializedColumn added to.
    excludes: Optional list of strings containing features, predictions, or
      labels to exclude from materialization.

  Returns:
    PCollection of Extracts
  """
  return extracts | 'MaterializeFeatures' >> beam.Map(_MaterializeFeatures,
                                                      excludes=excludes)
