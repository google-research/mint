# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build model from model config."""

from mint.core import fact_model


def _build_fact_model(model_config, is_training):
  model = fact_model.FACTModel(model_config.fact_model, is_training)
  return model


MODEL_BUILDER_MAP = {
    'fact_model': _build_fact_model,
}


def build(model_config, is_training):
  """Build model based on model_config."""
  model_type = model_config.WhichOneof('model')
  build_func = MODEL_BUILDER_MAP[model_type]
  return build_func(model_config, is_training)
