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
