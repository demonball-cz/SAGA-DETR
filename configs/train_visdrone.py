import os

from torch import optim

from datasets.coco import CocoDetection
from optimizer import param_dict
from transforms import presets


def env_int(name, default):
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def env_float(name, default):
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def env_path(name, default=None):
    value = os.getenv(name)
    return default if value in (None, "") else value


num_epochs = env_int("SAGA_EPOCHS", 12)
batch_size = env_int("SAGA_BATCH_SIZE", 1)
num_workers = env_int("SAGA_NUM_WORKERS", 4)
pin_memory = True
print_freq = env_int("SAGA_PRINT_FREQ", 50)
starting_epoch = 0
max_norm = 0.1

output_dir = env_path("SAGA_OUTPUT_DIR")
find_unused_parameters = False

coco_path = env_path("SAGA_COCO_PATH", "data/visdrone_coco")
train_transform = presets.detr
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train2017",
    ann_file=f"{coco_path}/annotations/instances_train2019.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val2017",
    ann_file=f"{coco_path}/annotations/instances_val2019.json",
    transforms=None,
)

model_path = env_path(
    "SAGA_MODEL_CONFIG",
    "configs/saga_detr/saga_detr_resnet50_visdrone.py",
)

resume_from_checkpoint = env_path("SAGA_RESUME")

learning_rate = env_float("SAGA_LR", 1e-4)
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(
    milestones=[env_int("SAGA_LR_DROP", 10)],
    gamma=0.1,
)

param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)
