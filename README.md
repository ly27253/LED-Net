# LED-Net Setup Guide

## 1. Environment Setup

This project is built based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation). To set up the environment, please follow the official installation guide:

- [MMsegmentation Installation Guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation)

## 2. Dataset

In addition to the configuration files for public datasets such as Cityscapes, ADE, COCO, and CamVid, we have written custom dataset loading scripts to accommodate our proprietary dataset, *Apple Branch Seg data*. The directory structure of the dataset is as follows:

Apple Branch Seg data ├── JPEGImages ├── SegmentationClassPNG ├── train.txt └── val.txt


- **JPEGImages**: Contains the original images.
- **SegmentationClassPNG**: Contains the segmentation labels.
- **train.txt** and **val.txt**: Define the training and validation data splits.

To use this dataset, you need to update the dataset path and directory information in the `../configs/_base_/datasets/pascal_voc12.py` configuration file. Additionally, modify the `classes` and `palette` in the `../mmseg/datasets/voc.py` file according to the dataset type and image file extensions.

## 3. Model Training

For training the LED-Net model, use the following configuration file:
../configs/LED_Net/LEDNet_80k_cityscapes-1024x1024.py

To specify the work directory for saving logs and models, use this parameter:
-work-dir', default="../LEDNet_fordata_11g07", help='the dir to save logs and models'

Adjust other parameters according to the provided instructions.

## 4. Model Testing

To test the model, use the following configuration and checkpoint settings:

- **Config file**: `../configs/LED_Net/LEDNet_80k_cityscapes-1024x1024.py`
- **Checkpoint file**: `../lednet_fordata_11g15/iter_80000.pth`

You can download the pretrained model checkpoint `iter_80000.pth` from [this link](#).

## 5. Model Performance (FLOPs) Testing

To test the model's computational cost (FLOPs), use the script `get_flops.py` and set the appropriate parameters.

## 6. Inference Speed Benchmarking

To test the inference speed, use the script `benchmark.py` and configure the parameters accordingly.

