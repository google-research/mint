# Mint: Multi-Modal Content Creation Infrastructure.

## Overview

This package contains the basic building blocks to a multi-modal video
understanding model, example audio-motion model and the infrastructure to train
such model. It supports:

*  BERT-style self-supervised and supervised training
*  GPT-style self-supervised and supervised training
*  Basic multi-headed attention layer

Multi-Modal Video Understanding Model Proposal: go/multi-modal-video-understanding

## Multi-Modal Video Understanding Model
The models in this package are all implemented using tensorflow keras and the
training pipeline is eager mode compatible.
### Model Building Blocks
*  TransformerModel with trainable positional encoding
*  InputEmbeddingLayer
*  OutputEmbeddingLayer
*  AttentionLayer


## Training Infrastructure
*  XManager GPU/TPU launching script
*  Eager train/eval functions
*  BERT Style on-line mask and attention mask creation
*  GPT Style shift look ahead mask support

## Getting Started
The launcher will automatically build everything needed, and then invoke the
borg files to bring up the train, eval, and tensorboard/mldash jobs. See
`scripts/*.sh` files for example commands of running the launcher.

### Build Multi-Modal Modal
*  Inherit from the MultiModalModel
*  Add modal specific TransformerModel
