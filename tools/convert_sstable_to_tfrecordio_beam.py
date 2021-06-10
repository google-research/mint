"""Binary to convert SSTable dataset to TFRecord dataset."""
from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.io import tfrecordio
import tensorflow as tf

from google3.pipeline.flume.py import runner as flume_runner
from google3.pipeline.flume.py.io import sstableio

FLAGS = flags.FLAGS

flags.DEFINE_string('input_sstable', '', 'Input sstables')
flags.DEFINE_string('output_filebase', '', 'Output root directory')
flags.DEFINE_integer('num_shard', 20, 'Number of shards.')


class ExtractTFExample(beam.DoFn):
  """Generates TFExample."""

  def process(self, kv):
    _, ex = kv
    yield ex.SerializeToString()


def main(argv):
  del argv  # Unused.

  with beam.Pipeline(runner=flume_runner.FlumeRunner()) as p:
    _ = (
        p
        | 'ReadSSTable' >> sstableio.ReadFromSSTable(
            filepattern=FLAGS.input_sstable,
            value_coder=beam.coders.ProtoCoder(tf.train.Example))
        | 'Extract example' >> beam.ParDo(ExtractTFExample())
        | 'Write to SSTable' >> tfrecordio.WriteToTFRecord(
            FLAGS.output_filebase,
            file_name_suffix='.tfrecord',
            num_shards=FLAGS.num_shard,
            coder=beam.coders.BytesCoder()))


if __name__ == '__main__':
  app.run(main)
