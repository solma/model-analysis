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
"""Writer types."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.types_compat import NamedTuple, Text

# A writer is a PTransform that takes Evaluation output as input and
# serializes the associated PCollections of data to a sink.
Writer = NamedTuple(
    'Writer',
    [
        ('stage_name', Text),
        # PTransform Evaluation -> PDone
        ('ptransform', beam.PTransform)
    ])


@beam.ptransform_fn
@beam.typehints.with_input_types(evaluator.Evaluation)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def Write(evaluation, key,
          ptransform):
  """Writes given Evaluation data using given writer PTransform.

  Args:
    evaluation: Evaluation data.
    key: Key for Evaluation output to write. It is valid for the key to not
      exist in the Evaluation dict (in which case the write is a no-op).
    ptransform: PTransform to use for writing.

  Raises:
    ValueError: If Evaluation is empty. The key does not need to exist in the
      Evaluation, but the Evaluation must not be empty.

  Returns:
    beam.pvalue.PDone.
  """
  if not evaluation:
    raise ValueError('Evaluation cannot be empty')
  if key in evaluation:
    return evaluation[key] | ptransform
  return beam.pvalue.PDone(list(evaluation.values())[0].pipeline)
