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
"""Init module for TensorFlow Model Analysis on notebook."""


from tensorflow_model_analysis import view
# pylint: disable=unused-import
from tensorflow_model_analysis import evaluators
from tensorflow_model_analysis import extractors
from tensorflow_model_analysis import slicer
from tensorflow_model_analysis import writers
from tensorflow_model_analysis.api import tfma_unit as test

from tensorflow_model_analysis.api.model_eval_lib import default_eval_shared_model
from tensorflow_model_analysis.api.model_eval_lib import default_evaluators
from tensorflow_model_analysis.api.model_eval_lib import default_extractors
from tensorflow_model_analysis.api.model_eval_lib import default_writers
from tensorflow_model_analysis.api.model_eval_lib import EvalConfig
from tensorflow_model_analysis.api.model_eval_lib import EvalResult
from tensorflow_model_analysis.api.model_eval_lib import ExtractAndEvaluate
from tensorflow_model_analysis.api.model_eval_lib import ExtractEvaluateAndWriteResults
from tensorflow_model_analysis.api.model_eval_lib import InputsToExtracts
from tensorflow_model_analysis.api.model_eval_lib import load_eval_result
from tensorflow_model_analysis.api.model_eval_lib import load_eval_results
from tensorflow_model_analysis.api.model_eval_lib import make_eval_results
from tensorflow_model_analysis.api.model_eval_lib import multiple_data_analysis
from tensorflow_model_analysis.api.model_eval_lib import multiple_model_analysis
from tensorflow_model_analysis.api.model_eval_lib import run_model_analysis
from tensorflow_model_analysis.api.model_eval_lib import WriteResults

from tensorflow_model_analysis.constants import ANALYSIS_KEY
from tensorflow_model_analysis.constants import DATA_CENTRIC_MODE
from tensorflow_model_analysis.constants import FEATURES_PREDICTIONS_LABELS_KEY
from tensorflow_model_analysis.constants import INPUT_KEY
from tensorflow_model_analysis.constants import METRICS_KEY
from tensorflow_model_analysis.constants import MODEL_CENTRIC_MODE
from tensorflow_model_analysis.constants import PLOTS_KEY

from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model import exporter

from tensorflow_model_analysis.post_export_metrics import post_export_metrics

from tensorflow_model_analysis.types import EvalSharedModel
from tensorflow_model_analysis.types import Extracts
from tensorflow_model_analysis.types import FeaturesPredictionsLabels
from tensorflow_model_analysis.types import TensorType
from tensorflow_model_analysis.types import TensorTypeMaybeDict

from tensorflow_model_analysis.version import VERSION_STRING

def _jupyter_nbextension_paths():
  return [{
      'section': 'notebook',
      'src': 'static',
      'dest': 'tfma_widget_js',
      'require': 'tfma_widget_js/extension'
  }]
