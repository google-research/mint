# AI Choreographer: Music Conditioned 3D Dance Generation with AIST++ [ICCV-2021].

## Overview

This package contains the model implementation and training infrastructure of
our AI Choreographer. 

## Setup
```
conda create -n mint python=3.7
conda activate mint
conda install protobuf numpy
pip install tensorflow absl-py tensorflow-datasets

sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
pip install tensorflow-graphics tensorflow-graphics-gpu

git clone https://github.com/arogozhnikov/einops /tmp/einops
cd /tmp/einops/ && pip install .
```

```
# complie protocols
protoc ./mint/protos/*.proto
# run training
python trainer.py --config_path ./configs/fact_v5_deeper_t10_cm12.config --model_dir ./checkpoints
```

## Training Infrastructure
* Orbit trainer

## Citation

```bibtex
@inproceedings{li2021dance,
  title={AI Choreographer: Music Conditioned 3D Dance Generation with AIST++},
  author={Ruilong Li and Shan Yang and David A. Ross and Angjoo Kanazawa},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year = {2021}
}
```
