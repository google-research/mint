#!/bin/bash
source gbash.sh || exit

DEFINE_string job_name_prefix \
  "mint_test" \
  "Experiment name."

DEFINE_string model_dir "" \
  "When running eval job only specify the model dir."

DEFINE_string config_path \
  "third_party/py/mint/configs/fact_v5_deeper_t10_cm12.config" \
  "The config file path."

DEFINE_int num_train_steps 1000000 \
  "Total number of train steps."

DEFINE_int warmup_steps 1000 \
  "Total number of warmup steps."

DEFINE_string gpu_cell \
  "jn" \
  "The XManager cell to run on."

DEFINE_string tpu_cell \
  "jn" \
  "The XManager cell to run on."

DEFINE_enum tpu_type "dragonfish" --enum="jellyfish,dragonfish"\
  "The TPU type to run on."

DEFINE_string tpu_topology "4x4" \
  "TPU topology."

DEFINE_int num_gpus 4 \
  "Number of gpus to use in the training worker."

DEFINE_enum job_type "trainval" --enum="trainval,train,val" ""

DEFINE_string peace_mdb "visual-dynamics" \
  "The mdb group to use for XManager resource allocation."

DEFINE_enum train_strategy "gpu" --enum="gpu,tpu" ""

DEFINE_double grad_clip_norm 0. \
  "Gradient clip to norm."

# Launch command:
gbash::init_google "$@"

GOOGLE3_DIR="$(gbash::google3_enclosing_path .)"

MODEL_BASE_DIR="/cns/is-d/home/${USER}/mint/exp"

JOB_NAME="${FLAGS_job_name_prefix}""_$(date +'%F')"
MODEL_DIR="${MODEL_BASE_DIR}""/${JOB_NAME}"

if [[ -n "$FLAGS_model_dir" ]]; then
  MODEL_DIR="${FLAGS_model_dir}"
fi

echo "$JOB_NAME"
echo "$MODEL_DIR"

/google/bin/releases/xmanager/cli/xmanager.par launch \
"${GOOGLE3_DIR}/third_party/py/mint/google/xm_launch.py" -- \
--job_name="${JOB_NAME}" \
--model_dir="${MODEL_DIR}" \
--tpu_type="${FLAGS_tpu_type}" \
--tpu_topology="${FLAGS_tpu_topology}" \
--num_train_gpus="${FLAGS_num_gpus}" \
--train_strategy="${FLAGS_train_strategy}" \
--config_path="${FLAGS_config_path}" \
--num_train_steps="${FLAGS_num_train_steps}" \
--warmup_steps="${FLAGS_warmup_steps}" \
--grad_clip_norm="${FLAGS_grad_clip_norm}" \
--gpu_cell="${FLAGS_gpu_cell}" \
--xm_resource_alloc="group:peace/${FLAGS_peace_mdb}" \
--xm_deployment_env=alphabet \
--noxm_monitor_on_launch
