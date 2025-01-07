_base_ = [
    # '../_base_/models/stdc.py', '../_base_/datasets/cityscapes.py',
    '../_base_/models/stdc.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 1024)
# crop_size = (128, 256)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=80000,
        by_epoch=False,
    )
]
# train_dataloader = dict(batch_size=12, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
train_dataloader = dict(batch_size=2, num_workers=1)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader
