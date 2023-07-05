# CGFormer
The official PyTorch implementation of the CVPR 2023 paper "Contrastive Grouping with Transformer for Referring Image Segmentation".

This paper first introduces learnable query tokens to represent objects and then alternately queries linguistic features and groups visual features into the query tokens for object-aware cross-modal reasoning. CGFormer achieves cross-level interaction by jointly updating the query tokens and decoding masks in every two consecutive layers. In addition, we introduce new splits on datasets for evaluating generalization for referring image segmentation models.

## Framework
<p align="center">
  <img src="image/framework.jpg" width="1000">
</p>

## Preparation

1. Environment
   - [PyTorch](www.pytorch.org) 
   - Other dependencies in `requirements.txt`
2. Datasets
   - The detailed instruction is in [prepare_datasets](data/READEME.md)
3. Pretrained weights
   - [Swin-Base-window12](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)

## Quick Start

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported. Besides, the evaluation only supports single-gpu mode.

To do training of CGFormer with 8 GPUs, run:

```
python -u train.py --config config/config.yaml
```

To do evaluation of CGFormer with 1 GPU, run:
```
CUDA_VISIBLE_DEVICES=0 python -u test.py \
      --config config/refcoco/config.yaml \
      --opts TEST.test_split val \
             TEST.test_lmdb path/val.lmdb
