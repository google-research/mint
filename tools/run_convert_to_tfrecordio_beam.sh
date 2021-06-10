#!/bin/bash
source gbash.sh || exit

DEFINE_string input_sstable ""\
  "The input sstables."

DEFINE_string output_filebase \
  "/cns/is-d/home/couchpotato/tf-mint/datasets/imagenet21k/mini_tfe" \
  "The output sstable filebase."

gbash::init_google "$@"

BIN=third_party/py/mint/tools/convert_sstable_to_tfrecordio_beam.par
rabbit --verifiable build -c opt \
${BIN}

function echo_and_run() {
  echo '$' "$@"
  "$@"
}

echo_and_run blaze-py3/bin/${BIN} \
  --borguser=couchpotato \
  --flume_exec_mode=BORG \
  --flume_borg_user_name=couchpotato \
  --flume_use_batch_scheduler=true \
  --input_sstable="${FLAGS_input_sstable}" \
  --output_filebase="${FLAGS_output_filebase}"
