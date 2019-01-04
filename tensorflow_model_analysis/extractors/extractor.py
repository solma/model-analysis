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

import apache_beam as beam
from tensorflow_model_analysis.types_compat import NamedTuple, Text

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
