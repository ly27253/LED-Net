#####################################################################
#  The code framework references MMsegmentation and DDRNet
#####################################################################

_base_ = [
    '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='LEDNet',
        in_channels=3,
        channels=32,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False),  # init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    decode_head=dict(
        type='LEDHead',
        in_channels=32 * 4,
        channels=64,
        dropout_ratio=0.,
        num_classes=2,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                # class_weight=class_weight,
                loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                # class_weight=class_weight,
                loss_weight=0.4),
        ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# train_dataloader = dict(batch_size=6, num_workers=4)
train_dataloader = dict(batch_size=2, num_workers=1)

iters = 80000

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]
