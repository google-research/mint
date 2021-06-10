"""Build model from model config."""

from mint.core import fact_model
from mint.core import vap_model


def _build_vap_model(model_config, is_training):
  model = vap_model.VAPModel(model_config.vap_model, is_training)
  return model


def _build_fact_model(model_config, is_training):
  model = fact_model.FACTModel(model_config.fact_model, is_training)
  return model


MODEL_BUILDER_MAP = {
    'vap_model': _build_vap_model,
    'fact_model': _build_fact_model,
}


def build(model_config, is_training):
  """Build model based on model_config."""
  model_type = model_config.WhichOneof('model')
  build_func = MODEL_BUILDER_MAP[model_type]
  return build_func(model_config, is_training)
