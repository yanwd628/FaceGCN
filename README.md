# FaceGCN (TCSVT)
---
FaceGCN: Structured Priors Inspired Graph 
Convolutional Networks for Blind Face Restoration

[paper](https://ieeexplore.ieee.org/document/10830527/) | [project](https://github.com/yanwd628/FaceGCN)

![](./imgs/overview.png)

The pipeline of FaceGCN for blind face restoration. The corrupted face (Input) in the feature domain is firstly constructed into a face graph with case-specific guidance from the Dynamic Adjacency Matrix Generator. Then, some Strip-Attention GCN Modules are stacked to finally produce the restored face (Output) benefiting from the captured joint local-nonlocal correlations among various facial feature components.


## Dependencies
+ Python 3.6
+ PyTorch >= 1.7.0
+ matplotlib
+ opencv
+ torchvision
+ numpy


## Datasets are provided [here](https://github.com/wzhouxiff/RestoreFormer?tab=readme-ov-file#preparations-of-dataset-and-models)


## Train and Test (based on [Basicsr](https://github.com/XPixelGroup/BasicSR))

    python facegcn/train.py -opt options/train/train_stage_1.yml --auto_resume
    python facegcn/train.py -opt options/train/train_stage_2.yml --auto_resume
    python facegcn/test.py -opt options/test/test_xxx.yml

**ps: the path configs should be changed to your own path**

Our pretrained model is available [Google Drive](https://drive.google.com/file/d/1vqkWQX0Byd2kVBr9MwqPN3oI1bSeEPy5/view?usp=sharing)

```
@ARTICLE{10830527,
  author={Yan, Weidan and Shao, Wenze and Zhang, Dengyin and Xiao, Liang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={FaceGCN: Structured Priors Inspired Graph Convolutional Networks for Blind Face Restoration}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Faces;Image restoration;Face recognition;Facial features;Degradation;Transformers;Correlation;Pipelines;Noise reduction;Generators;Blind Face Restoration;Graph Convolution Networks;Structured Priors;Strip-Attention Mechanism},
  doi={10.1109/TCSVT.2025.3526841}}
```


