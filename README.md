# LED-Net Setup Guide

## 1. Environment Setup  
This project is based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation). To set up the environment, please follow the official installation guide:  
- [MMsegmentation Installation Guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation)

## 2. Dataset  
In addition to the configuration files for public datasets like Cityscapes, ADE, COCO, and CamVid, we have written custom dataset loading scripts to support our proprietary *Apple Branch Seg data* dataset. The directory structure is as follows:  

## Dataset Structure

In addition to the configuration files for public datasets like Cityscapes, ADE, COCO, and CamVid, we have written custom dataset loading scripts to support our proprietary *Apple Branch Seg data* dataset. The directory structure is as follows:

<pre>
├── Apple Branch Seg data 
        ├── JPEGImages              # Contains original images 
        ├── SegmentationClassPNG    # Contains segmentation labels 
        ├── train.txt               # Training data split 
        └── val.txt                 # Validation data split
</pre>


- **JPEGImages**: Original images of the dataset.  
- **SegmentationClassPNG**: Corresponding segmentation labels in PNG format.  
- **train.txt** & **val.txt**: Files that define the training and validation data splits.

To use this dataset, update the dataset path and directory structure in the `../configs/_base_/datasets/pascal_voc12.py` configuration file. Additionally, adjust the `classes` and `palette` in the `../mmseg/datasets/voc.py` file according to your dataset's types and image file extensions.

## 3. Model Training  
For training the LED-Net model, use the following configuration file:  
`../configs/LED_Net/LEDNet_80k_cityscapes-1024x1024.py`  

Set the work directory to save logs and models:  
`--work-dir`, default is `../LEDNet_fordata_11g07`, help='Directory to save logs and models'  

Adjust other parameters as per the provided instructions.

## 4. Model Testing  
To test the trained model, use the following configuration and checkpoint settings:  
- **Config file**: `../configs/LED_Net/LEDNet_80k_cityscapes-1024x1024.py`  
- **Checkpoint file**: `../lednet_fordata_11g15/iter_80000.pth`  

You can download the pretrained model checkpoint `iter_80000.pth` from [this link](#).

## 5. Model Performance (FLOPs) Testing  
To evaluate the model's computational cost (FLOPs), use the `get_flops.py` script and set the appropriate parameters.

## 6. Inference Speed Benchmarking  
To benchmark the inference speed, use the `benchmark.py` script and adjust the settings as needed.
