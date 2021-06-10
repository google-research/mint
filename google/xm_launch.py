r"""XM Launcher for training/evaluation of Conditional Motion Prediction model.

Note that the defaults are set for YFCC dataset (Pathak subset) dataset.

Example command line:
-------------------
google_xmanager launch \
third_party/py/mint/google/xm_launch.py -- \
--job_name="${USER}_mint" \
--model_dir="/cns/ym-d/home/${USER}/mint" \
--xm_resource_alloc="group:peace/visual-dynamics"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging

from google3.learning.deepmind.python.adhoc_import import binary_import
from google3.learning.deepmind.xmanager import hyper
from google3.learning.deepmind.xmanager2.client import google as xm
from google3.pyglib import gfile

with binary_import.AutoGoogle3():
  # pylint: disable=g-import-not-at-top
  from google3.learning.brain.frameworks.xmanager import xm_helper

TPU_TYPES = ['jellyfish', 'dragonfish']
TRAIN_STRATEGY = ['tpu', 'gpu']

FLAGS = flags.FLAGS

flags.DEFINE_string('config_path', None, 'Path to pipeline config.')
flags.DEFINE_string(
    'gpu_cell', None, 'Cell to run jobs in. When unspecified, '
    'automatically picks a cell and this is '
    'highly recommended.')
flags.DEFINE_string(
    'tpu_cell', None, 'Cell to run tpu job. When unspecified, '
    'defaults to --cell and is recommended.')
flags.DEFINE_string('service_tier', 'PROD', 'Valid Options are - [PROD, BATCH, '
                    'FREEBIE].')
flags.DEFINE_string('tpu_topology', '2x2', 'TPU topology.')
flags.DEFINE_integer('num_train_gpus', 8, 'Number of GPUs for training with '
                     'mirrored strategy.')
flags.DEFINE_enum('train_strategy', TRAIN_STRATEGY[0], TRAIN_STRATEGY,
                  'Whether to train with TPUs or Mirrored GPUs.')
flags.DEFINE_enum('tpu_type', TPU_TYPES[1], TPU_TYPES, 'TPU type')
flags.DEFINE_string('job_name', None, 'Name for the job.')
flags.DEFINE_string('model_dir', None, 'Path to write checkpoints and event '
                    'files.')
flags.DEFINE_integer('num_train_steps', 50000, 'Number of training steps')
flags.DEFINE_integer('warmup_steps', 100, 'Number of training steps')
flags.DEFINE_float('grad_clip_norm', 0., 'Clip gradient by norm.')
flags.mark_flag_as_required('job_name')
flags.mark_flag_as_required('model_dir')


def _copy_config_to_model_dir():
  """Copies config file to model directory."""
  try:
    gfile.MakeDirs(FLAGS.model_dir)
    new_config_path = os.path.join(FLAGS.model_dir,
                                   os.path.basename(FLAGS.config_path))
    gfile.Copy(FLAGS.config_path, new_config_path, overwrite=True)
  except gfile.GOSError:
    logging.fatal('Failed to copy config from %s to %s.', FLAGS.config_path,
                  FLAGS.model_dir)
  return new_config_path


def sweep_lr_only(model_dir):
  """Returns a sweep with the best set of hyperparams found so far.

  Args:
    model_dir: base model directory.

  Returns:
    hyperparams iterator for jobs.
  """
  sweep_config = hyper.product([
      hyper.sweep('head_initializer', ['glorot_uniform']),
      hyper.sweep('weight_decay', [0]),
      hyper.sweep('initial_learning_rate', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
  ])
  model_dirs = []
  for config in sweep_config:
    model_dirs.append(
        os.path.join(
            model_dir,
            'lr:{:.1e}_init:{}'.format(config['initial_learning_rate'],
                                       config['head_initializer'])))
  return hyper.zipit([sweep_config, hyper.sweep('model_dir', model_dirs)])


def sweep_lr_head_initializer(model_dir):
  """Returns hyperparameter sweep for learning rate and head initializer.

  Args:
    model_dir: base model directory.

  Returns:
    hyperparams iterator for jobs.
  """
  sweep_config = hyper.product([
      hyper.sweep('head_initializer', ['glorot_uniform', 'he_normal', 'zeros']),
      hyper.sweep('weight_decay', [0]),
      hyper.sweep('initial_learning_rate', [1e-2, 1e-3, 1e-4])
  ])
  model_dirs = []
  for config in sweep_config:
    model_dirs.append(
        os.path.join(
            model_dir,
            'lr:{:.1e}_init:{}'.format(config['initial_learning_rate'],
                                       config['head_initializer'])))
  return hyper.zipit([sweep_config, hyper.sweep('model_dir', model_dirs)])


def sweep_lr_weight_decay(model_dir):
  """Returns hyperparameter sweep for learning rate and weight decay.

  Args:
    model_dir: base model directory.

  Returns:
    hyperparams iterator for jobs.
  """
  sweep_config = hyper.product([
      hyper.sweep('weight_decay', [1e-4, 1e-5, 1e-6]),
      hyper.sweep('initial_learning_rate', [1e-2, 1e-3, 1e-4])
  ])
  model_dirs = []
  for config in sweep_config:
    model_dirs.append(
        os.path.join(
            model_dir,
            'lr:{:.1e}_l2:{:.1e}'.format(config['initial_learning_rate'],
                                         config['weight_decay'])))
  return hyper.zipit([sweep_config, hyper.sweep('model_dir', model_dirs)])


def get_service_tier():
  if FLAGS.service_tier == 'PROD':
    return xm.ServiceTier.PROD
  elif FLAGS.service_tier == 'BATCH':
    return xm.ServiceTier.BATCH
  elif FLAGS.service_tier == 'FREEBIE':
    return xm.ServiceTier.FREEBIE
  else:
    return xm.ServiceTier.AUTO


def build_tpu_jobs(args, tpu_topology, name, coordinator_cell, worker_cell):
  """Builds a TPU work unit."""

  scheduling = dict(max_task_failures=15, max_per_task_failures=-1)
  overrides = xm.BorgOverrides(scheduling=scheduling)
  # TODO(b/147832565) Remove once TF2 can gracefully handle preemption
  if tpu_topology == '4x2':
    overrides.scheduling.policy = 'BEST_EFFORT'
    overrides.scheduling.size = 1
    overrides.tasks_per_host = 1
    overrides.params.enable_partial_slice = True

  exec_coordinator = xm.BuildTarget(
      '//third_party/py/mint:trainer',
      name=name + '_tpu_coordinator',
      args=args,
      runtime=xm.Borg(
          cell=coordinator_cell,
          service_tier=get_service_tier(),
          overrides=overrides,
          logs_read_access_roles=['all'],
          requirements=xm.Requirements(
              ram=20 * xm.GiB, tmp_ram_fs_size=256 * xm.MiB)),
  )

  runtime_tpu_worker = xm.Borg(
      cell=worker_cell,
      service_tier=get_service_tier(),
      requirements=xm.Requirements(topology=xm.TpuTopology(tpu_topology)),
      logs_read_access_roles=['all'],
  )

  return xm_helper.build_tpu_jobs(
      name=name + '_tpu_worker',
      coordinator=exec_coordinator,
      tpu_runtime=runtime_tpu_worker,
      tpu_platform=xm.Platform.from_str(FLAGS.tpu_type),
      brain_port_name='',
      args={
          'brain_rpc_layer': 'grpc',
      })


def build_gpu_job(args, name, num_gpus, cell, gpu_types, binary):
  """Builds a GPU work unit."""
  overrides = xm.BorgOverrides(xm_pass_arguments=False)
  exec_evaluator = xm.BuildTarget(
      binary,
      name=name + '_gpu_worker',
      args=args,
      platform=xm.Platform.GPU,
      runtime=xm.Borg(
          cell=cell,
          service_tier=get_service_tier(),
          overrides=overrides,
          logs_read_access_roles=['all'],
          requirements=xm.Requirements(
              gpu=num_gpus,
              cpu=num_gpus * 8,
              ram=num_gpus * 20 * (2**30),
              tmp_ram_fs_size=256 * xm.MiB,
              gpu_types=gpu_types)))
  return exec_evaluator


def add_monitoring(experiment):
  """Adds tensorboard and mldash monitoring to the experiment."""
  cell = xm.Borg.default_cell_selector()
  experiment = xm.WithTensorBoard(experiment, FLAGS.model_dir)
  experiment = xm_helper.WithMLDash(
      experiment,
      FLAGS.model_dir,
      runtime=xm.Borg(cell=cell, service_tier=get_service_tier()),
      # Give MLDash 30 mins to save the final summaries.
      termination_delay_secs=1800)
  return experiment


def build_experiment(train_args, eval_jobs_args, model_dir):
  """Create the jobs and return the constructed experiment."""
  coordinator_cell = xm.Borg.default_cell_selector()
  gpu_cell = xm.Borg.default_cell_selector()
  gpu_types = xm.Borg.gpu_selector([xm.GpuType.P100])
  if FLAGS.tpu_cell:
    coordinator_cell = xm.Borg.cell_selector([FLAGS.tpu_cell])
  if FLAGS.gpu_cell:
    gpu_cell = xm.Borg.cell_selector([FLAGS.gpu_cell])
  worker_cell = FLAGS.tpu_cell or coordinator_cell

  if FLAGS.train_strategy == TRAIN_STRATEGY[0]:
    train_executables = build_tpu_jobs(
        train_args,
        tpu_topology=FLAGS.tpu_topology,
        name='training',
        coordinator_cell=coordinator_cell,
        worker_cell=worker_cell)
  else:
    train_binary = ('//third_party/py/mint:trainer')
    train_executables = tuple([
        build_gpu_job(
            train_args,
            'training',
            FLAGS.num_train_gpus,
            gpu_cell,
            gpu_types,
            binary=train_binary)
    ])

  eval_executables = []
  eval_binary = ('//third_party/py/mint:evaluator')
  for args in eval_jobs_args:
    eval_executables.append(
        build_gpu_job(
            args,
            'eval_' + args['eval_prefix'],
            1,
            gpu_cell,
            gpu_types,
            binary=eval_binary))
  experiment = xm.ParallelExecutable(
      train_executables + tuple(eval_executables), name=FLAGS.job_name)
  hyperparams = sweep_lr_only(model_dir)
  experiment = xm.ParameterSweep(experiment, hyperparams)
  experiment = add_monitoring(experiment)
  return experiment


def main(_):
  """Launch the experiment using the arguments from the command line."""
  description = xm.ExperimentDescription(
      'Experiment: %s' % FLAGS.job_name, tags=['mint'])

  config_path = _copy_config_to_model_dir()
  train_args = {
      'train_strategy': FLAGS.train_strategy,
      'model_dir': FLAGS.model_dir,
      'config_path': config_path,
      'steps': FLAGS.num_train_steps,
      'warmup_steps': FLAGS.warmup_steps,
      'grad_clip_norm': FLAGS.grad_clip_norm,
      'xprof_port': '%port_xprof%'
  }

  eval_args = [{
      'eval_prefix': 'train',
      'config_path': config_path,
      'model_dir': FLAGS.model_dir,
  }, {
      'eval_prefix': 'validation',
      'config_path': config_path,
      'model_dir': FLAGS.model_dir,
  }]

  experiment = build_experiment(train_args, eval_args, FLAGS.model_dir)
  xm.launch_experiment(description, experiment)


if __name__ == '__main__':
  app.run(main)
