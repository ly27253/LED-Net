# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type

import os
import mmcv
import torch
import torch.nn as nn

from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
from mmseg.apis import inference_model, init_model
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
from mmseg.visualization import SegLocalVisualizer

'''
/media/g/mydata/branch_ly/JPEGImages/000001.png
/home/g/segmentation/0705_seg/mmseg_06g06/configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py
/media/g/mydata/results/ddr_fordata_06g19/iter_80000.pth
--gt_mask /media/g/mydata/branch_ly/SegmentationClassPNG/000001.png
'''

'''
/media/g/mydata/branch_ly/JPEGImages/000001.png
/home/g/segmentation/0705_seg/mmseg_06g06/configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py
/media/g/mydata/results/ddr_fordata_06g19/iter_80000.pth
--gt_mask /media/g/mydata/out.png
'''

class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


def visualize(args, model, recorder, result):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=0.5)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])

    image = mmcv.imread(args.img, 'color')

    seg_visualizer.add_datasample(
        name='predict',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        wait_time=0,
        out_file=None,
        show=False)

    # add feature map to wandb visualizer
    for i in range(len(recorder.data_buffer)):
        feature = recorder.data_buffer[i][0]  # remove the batch
        drawn_img = seg_visualizer.draw_featmap(
            feature, image, channel_reduction='select_max')
        seg_visualizer.add_image(f'feature_map{i}', drawn_img)
        # 保存feature_map在本地
        mmcv.imwrite(mmcv.rgb2bgr(drawn_img), '/media/g/mydata/class1.png')


    if args.gt_mask:
        sem_seg = mmcv.imread(args.gt_mask, 'unchanged')
        sem_seg = torch.from_numpy(sem_seg)
        gt_mask = dict(data=sem_seg)
        gt_mask = PixelData(**gt_mask)
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = gt_mask

        seg_visualizer.add_datasample(
            name='gt_mask',
            image=image,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=False,
            wait_time=0,
            out_file=None,
            show=False)

    seg_visualizer.add_image('image', image)
    # mmcv.imwrite(mmcv.rgb2bgr(image), '/media/g/mydata/image.png')


def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gt_mask', default=None, help='Path of gt mask file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # show all named module in the model and use it in source list below
    for name, module in model.named_modules():
        print(name)

    # source = [
    #     'decode_head.fusion.stages.0.query_project.activate',
    #     'decode_head.context.stages.0.key_project.activate',
    #     'decode_head.context.bottleneck.activate'
    # ]

    # mmseg_06g06中ddrnet的decode_head
    # feature map 显示在zesty-water-3（https://wandb.ai/tcp-127-0-0-1-50163/mmseg_06g06-tools）
    # source = [
    #     'decode_head.head.0.activate',
    #     'decode_head.aux_head.0.activate',
    #     'decode_head.head_x1.0.activate',
    #     'decode_head.head_x2.0.activate'
    # ]

    source = [
        # 'backbone.aff1.local_att.0'
        # 'backbone.layer5_.spp_dw.1'
        'decode_head.head.0.activate'
    ]

    source = dict.fromkeys(source)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break

    with recorder:
        # test a single image, and record feature map to data_buffer
        result = inference_model(model, args.img)

    visualize(args, model, recorder, result)


if __name__ == '__main__':
    main()
