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
"""Extractor type."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy

import apache_beam as beam
from tensorflow_model_analysis import types
from tensorflow_model_analysis.types_compat import List, NamedTuple, Optional, Text

# Tag for the last extractor in list of extractors.
LAST_EXTRACTOR_STAGE_NAME = '<last-extractor>'

# An Extractor is a PTransform that takes Extracts as input and returns Extracts
# as output. A typical example is a PredictExtractor that receives an 'input'
# placeholder for input and adds additional 'features', 'labels', and
# 'predictions' extracts.
Extractor = NamedTuple(  # pylint: disable=invalid-name
    'Extractor',
    [
        ('stage_name', Text),
        # PTransform Extracts -> Extracts
        ('ptransform', beam.PTransform)
    ])


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(beam.typehints.Any)
def Filter(extracts,
           include = None,
           exclude = None):
  """Filters extracts to include/exclude specified keys.

  Args:
    extracts: PCollection of extracts.
    include: Keys to include in output.
    exclude: Keys to exclude from output.

  Returns:
    Filtered PCollection of Extracts.

  Raises:
    ValueError: If both include and exclude are used.
  """
  if include and exclude:
    raise ValueError('only one of include or exclude should be used.')

  def filter_extracts(extracts):  # pylint: disable=invalid-name
    """Filters extracts."""
    if not include and not exclude:
      return extracts
    if include:
      filtered = {}
      for key in extracts:
        if key in include:
          filtered[key] = extracts[key]
    if exclude:
      filtered = copy.copy(extracts)
      for key in exclude:
        del filtered[key]
    return filtered

  return extracts | beam.Map(filter_extracts)
