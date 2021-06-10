"""Text preprocessing utility functions.

These functions are largely obtained from
google3/third_party/tensorflow_models/google/nlp/data/pretrain_text_dataloader.py
"""


import google3

import tensorflow as tf
import tensorflow_text as tf_text

_CLS_TOKEN = b"[CLS]"
_SEP_TOKEN = b"[SEP]"
_MASK_TOKEN = b"[MASK]"
_NUM_OOV_BUCKETS = 1
# Accounts for [CLS] and 2 x [SEP] tokens
_NUM_SPECIAL_TOKENS = 3


class BertPretrainTextPreprocessor():
  """A class to preprocess text datasets for the BERT pretraining task."""

  def __init__(self, text_preprocessor_config):
    """Inits `BertPretrainTextPreprocessor` class.

    Args:
      text_preprocessor_config: config file with the correct fields
    """
    self.params = text_preprocessor_config
    self.seq_length = self.params.seq_length
    self.max_predictions_per_seq = self.params.max_predictions_per_seq
    self.masking_rate = self.params.masking_rate
    self.use_whole_word_masking = self.params.use_whole_word_masking
    self.deterministic = self.params.deterministic

    lookup_table_init = tf.lookup.TextFileInitializer(
        text_preprocessor_config.vocab_file_path,
        key_dtype=tf.string,
        key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
    self.vocab_lookup_table = tf.lookup.StaticVocabularyTable(
        lookup_table_init,
        num_oov_buckets=_NUM_OOV_BUCKETS,
        lookup_key_dtype=tf.string)

    self.cls_token = self.vocab_lookup_table.lookup(tf.constant(_CLS_TOKEN))
    self.sep_token = self.vocab_lookup_table.lookup(tf.constant(_SEP_TOKEN))
    self.mask_token = self.vocab_lookup_table.lookup(tf.constant(_MASK_TOKEN))

    # -_NUM_OOV_BUCKETS to offset unused OOV bucket.
    self.vocab_size = self.vocab_lookup_table.size() - _NUM_OOV_BUCKETS

  @tf.function
  def bert_preprocess_text(self, input_text):
    """Parses raw tensors into a dict of tensors to be consumed by the model.

    This function only tokenizes and masks for the MLM objective. Currently
    it does not support next sentence prediction.
    Args:
      input_text: a list of string tensor.

    Returns:
      model_inputs: dictionary of model inputs for one string tensor. Keys and
      shapes of tensors are:
        "input_word_ids": [1, seq_length]
        "input_type_ids": [1, seq_length]
        "masked_lm_positions": [1, seq_length]
        "masked_lm_ids": [1, max_predictions_per_seq]
        "masked_lm_weights": [1, max_predictions_per_seq]
        "input_mask": [1, max_predictions_per_seq]
    }
    """
    joint_text = tf.strings.reduce_join(input_text, separator="\n")
    # Tokenize segments
    tokenizer = tf_text.BertTokenizer(
        self.vocab_lookup_table, token_out_type=tf.int64)
    if self.use_whole_word_masking:
      # tokenize the segments which should have the shape:
      # [num_sentence, (num_words), (num_wordpieces)]
      tokenized_text = tokenizer.tokenize(joint_text)
    else:
      # tokenize the segments and merge out the token dimension so that each
      # segment has the shape: [num_sentence, (num_wordpieces)]
      tokenized_text = tokenizer.tokenize(joint_text).merge_dims(-2, -1)

    # Truncate inputs
    trimmer = tf_text.WaterfallTrimmer(
        self.seq_length - _NUM_SPECIAL_TOKENS, axis=-1)
    truncated_text = trimmer.trim([tokenized_text])

    # Combine segments, get segment ids and add special tokens
    # TODO(anagrani) do we want to remove CLS?
    combined_text, text_ids = tf_text.combine_segments(
        truncated_text,
        start_of_sequence_id=self.cls_token,
        end_of_segment_id=self.sep_token)

    # Dynamic masking
    item_selector = tf_text.RandomItemSelector(
        self.max_predictions_per_seq,
        selection_rate=self.masking_rate,
        unselectable_ids=[self.cls_token, self.sep_token],
        shuffle_fn=(tf.identity if self.params.deterministic else None))
    values_chooser = tf_text.MaskValuesChooser(
        vocab_size=self.vocab_size, mask_token=self.mask_token)
    masked_input_ids, masked_lm_positions, masked_lm_ids = (
        tf_text.mask_language_model(
            combined_text,
            item_selector=item_selector,
            mask_values_chooser=values_chooser,
        ))

    # Pad out to fixed shape and get input mask.
    seq_lengths = {
        "input_word_ids": self.seq_length,
        "input_type_ids": self.seq_length,
        "masked_lm_positions": self.max_predictions_per_seq,
        "masked_lm_ids": self.max_predictions_per_seq,
    }
    model_inputs = {
        "input_word_ids": masked_input_ids,
        "input_type_ids": text_ids,
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_ids": masked_lm_ids,
    }
    padded_inputs_and_mask = tf.nest.map_structure(tf_text.pad_model_inputs,
                                                   model_inputs, seq_lengths)
    model_inputs = {
        k: padded_inputs_and_mask[k][0] for k in padded_inputs_and_mask
    }
    model_inputs["masked_lm_weights"] = (
        padded_inputs_and_mask["masked_lm_ids"][1])
    model_inputs["input_mask"] = padded_inputs_and_mask["input_word_ids"][1]

    return model_inputs
