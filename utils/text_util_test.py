"""Tests for mint.utils.text_util."""


from mint.protos import preprocessor_pb2
from mint.utils import text_util
import tensorflow as tf
from google3.testing.pybase import googletest

_TEST_VOCAB_PATH = "third_party/tensorflow_models/google/nlp/data/testdata/vocab.txt"


class TextUtilTest(tf.test.TestCase):

  def test_bertpreprocess_text(self):
    text_preprocessor = preprocessor_pb2.Preprocessor()
    text_preprocessor.text_preprocessor.input_path = ""
    text_preprocessor.text_preprocessor.doc_batch_size = 2
    text_preprocessor.text_preprocessor.global_batch_size = 2
    text_preprocessor.text_preprocessor.is_training = False
    text_preprocessor.text_preprocessor.seq_length = 128
    text_preprocessor.text_preprocessor.max_predictions_per_seq = 5
    text_preprocessor.text_preprocessor.deterministic = True
    text_preprocessor.text_preprocessor.text_field_name = "text"
    text_preprocessor.text_preprocessor.vocab_file_path = _TEST_VOCAB_PATH

    # Example text
    text_dataset = [
        # doc 1
        "Satyricon je norveški black metal sastav osnovan 1991. godine u "
        "Oslu.\nSatyr" + " i Frost ključni su članovi sastava od 1993. godine.",
        "Prva tri " +
        " albuma sastava tipičan su primjer norveškog black metal glazbenog "
        "stila.",
        "Sastav " + "se " +
        "okrenuo od tog stila glazbe počevši sa svojim četvrtim albumom te "
        "je počeo " + "uključivati elemente hard rocka u svoju glazbu.",
        "Satyricon je prvi "
        "norveški " +
        "black metal sastav koji se pridružio multinacionalnoj diskografskoj "
        "kući " + "(EMI).",
        # doc 2
        "Stuart är en stad (city) i Martin County, i delstaten Florida, "
        "USA.",
        "Enligt" +
        "United States Census Bureau har staden en folkmängd på 15 718 "
        "invånare " + "(2011) och en landarea på 17,2 km².",
        "Stuart är huvudort i Martin "
        "County.",
        # doc 3
        "Hello there.",
        " What time is it?",
        "Who let the dogs out?",
        # doc 4
        "AT first I was afraid.",
        "I was petrified.",
        "I didn't know "
        "how to let you know\n",
    ]

    input_feature = tf.constant(text_dataset)

    text_datapreprocessor = text_util.BertPretrainTextPreprocessor(
        text_preprocessor.text_preprocessor)
    processed_features = text_datapreprocessor.bert_preprocess_text(
        input_feature)
    self.assertLen(processed_features, 6)
    self.assertAllEqual(
        tf.shape(processed_features["input_word_ids"]), [1, 128])
    self.assertIn("input_word_ids", processed_features)
    self.assertIn("input_mask", processed_features)
    self.assertIn("input_type_ids", processed_features)
    self.assertAllEqual(
        tf.shape(processed_features["masked_lm_positions"]), [1, 5])
    self.assertIn("masked_lm_positions", processed_features)
    self.assertIn("masked_lm_ids", processed_features)
    self.assertIn("masked_lm_weights", processed_features)

  def test_whole_word_masking(self):
    text_preprocessor = preprocessor_pb2.Preprocessor()
    text_preprocessor.text_preprocessor.input_path = ""
    text_preprocessor.text_preprocessor.doc_batch_size = 2
    text_preprocessor.text_preprocessor.global_batch_size = 2
    text_preprocessor.text_preprocessor.is_training = False
    text_preprocessor.text_preprocessor.seq_length = 128
    text_preprocessor.text_preprocessor.max_predictions_per_seq = 5
    text_preprocessor.text_preprocessor.deterministic = True
    text_preprocessor.text_preprocessor.text_field_name = "text"
    text_preprocessor.text_preprocessor.vocab_file_path = _TEST_VOCAB_PATH
    text_preprocessor.text_preprocessor.use_whole_word_masking = True
    text_preprocessor.text_preprocessor.masking_rate = 0.5

    wwm_dataset = [
        "floridaham",
        "hostization",
    ]

    input_feature = tf.constant(wwm_dataset)
    text_datapreprocessor = text_util.BertPretrainTextPreprocessor(
        text_preprocessor.text_preprocessor)
    processed_features = text_datapreprocessor.bert_preprocess_text(
        input_feature)

    # Verify that whole words have been selected for masking.
    # Read vocab
    vocab_raw = tf.io.read_file(_TEST_VOCAB_PATH)
    vocab = tf.strings.split(vocab_raw, "\n")
    # Gather ids and decode
    decoded_masked_inputs = tf.gather(vocab,
                                      processed_features["masked_lm_ids"])
    # Only expecting to select 0.5 * 2 = 1 words
    self.assertAllEqual(
        [[b"florida", b"##ham", b"[PAD]", b"[PAD]", b"[PAD]"]],
        decoded_masked_inputs)

if __name__ == "__main__":
  googletest.main()
