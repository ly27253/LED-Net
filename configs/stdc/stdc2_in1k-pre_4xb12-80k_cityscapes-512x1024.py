# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'  # noqa
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa
_base_ = './stdc2_4xb12-80k_cityscapes-512x1024.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint))))
