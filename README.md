# AI Choreographer: Music Conditioned 3D Dance Generation with AIST++ [ICCV-2021].

## Overview

This package contains the model implementation and training infrastructure of
our AI Choreographer. 

## Get started

#### Pull the code
```
git clone https://github.com/liruilong940607/mint --recursive
```
Note here `--recursive` is important as it will automatically clone the submodule ([orbit](https://github.com/tensorflow/models/tree/master/orbit)) as well.

#### Install dependencies
```
conda create -n mint python=3.7
conda activate mint
conda install protobuf numpy
pip install tensorflow absl-py tensorflow-datasets librosa

sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
pip install tensorflow-graphics tensorflow-graphics-gpu

git clone https://github.com/arogozhnikov/einops /tmp/einops
cd /tmp/einops/ && pip install . -U

git clone https://github.com/google/aistplusplus_api /tmp/aistplusplus_api
cd /tmp/aistplusplus_api && pip install -r requirements.txt && pip install . -U
```
Note if you meet environment conflicts about numpy, you can try with `pip install numpy==1.20`. 

#### Get the data
See the [website](https://google.github.io/aistplusplus_dataset/)

#### Get the checkpoint
Download from google drive [here](https://drive.google.com/drive/folders/17GHwKRZbQfyC9-7oEpzCG8pp_rAI0cOm?usp=sharing), and put them to the folder `./checkpoints/`

#### Run the code

1. complie protocols
```
protoc ./mint/protos/*.proto
```

2. preprocess dataset into tfrecord
```
python tools/preprocessing.py \
    --anno_dir="/mnt/data/aist_plusplus_final/" \
    --audio_dir="/mnt/data/AIST/music/" \
    --split=train
python tools/preprocessing.py \
    --anno_dir="/mnt/data/aist_plusplus_final/" \
    --audio_dir="/mnt/data/AIST/music/" \
    --split=testval
```

3. run training
```
python trainer.py --config_path ./configs/fact_v5_deeper_t10_cm12.config --model_dir ./checkpoints
```
Note you might want to change the `batch_size` in the config file if you meet OUT-OF-MEMORY issue.

4. run testing and evaluation
```
# caching the generated motions (seed included) to `./outputs`
python evaluator.py --config_path ./configs/fact_v5_deeper_t10_cm12.config --model_dir ./checkpoints
# calculate FIDs
python tools/calculate_scores.py
```


## Citation

```bibtex
@inproceedings{li2021dance,
  title={AI Choreographer: Music Conditioned 3D Dance Generation with AIST++},
  author={Ruilong Li and Shan Yang and David A. Ross and Angjoo Kanazawa},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year = {2021}
}
```
