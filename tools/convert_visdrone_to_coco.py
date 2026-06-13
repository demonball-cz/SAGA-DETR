import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VISDRONE_CATEGORIES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VisDrone DET annotations to COCO format.")
    parser.add_argument(
        "--train-root",
        default="/path/to/VisDrone2019-DET-train/VisDrone2019-DET-train",
    )
    parser.add_argument(
        "--val-root",
        default="/path/to/VisDrone2019-DET-val/VisDrone2019-DET-val",
    )
    parser.add_argument("--out-dir", default="data/visdrone_coco")
    return parser.parse_args()


def read_image_size(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    height, width = image.shape[:2]
    return width, height


def convert_split(root, split_name):
    root = Path(root)
    image_dir = root / "images"
    ann_dir = root / "annotations"
    image_paths = sorted(image_dir.glob("*.jpg"))
    categories = [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(VISDRONE_CATEGORIES)
    ]
    images = []
    annotations = []
    skipped = {"ignored_class": 0, "invalid_box": 0, "missing_annotation": 0}
    ann_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        width, height = read_image_size(image_path)
        images.append(
            {
                "id": image_id,
                "file_name": str(image_path.resolve()),
                "width": width,
                "height": height,
            }
        )
        txt_path = ann_dir / f"{image_path.stem}.txt"
        if not txt_path.exists():
            skipped["missing_annotation"] += 1
            continue

        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = [int(float(x)) for x in line.split(",") if x != ""]
                if len(fields) < 6:
                    skipped["invalid_box"] += 1
                    continue
                x, y, w, h, score, original_category = fields[:6]
                truncation = fields[6] if len(fields) > 6 else 0
                occlusion = fields[7] if len(fields) > 7 else 0

                if score <= 0 or original_category in (0, 11) or not (1 <= original_category <= 10):
                    skipped["ignored_class"] += 1
                    continue

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + w)
                y2 = min(height, y + h)
                clipped_w = x2 - x1
                clipped_h = y2 - y1
                if clipped_w <= 1 or clipped_h <= 1:
                    skipped["invalid_box"] += 1
                    continue

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": original_category - 1,
                        "bbox": [float(x1), float(y1), float(clipped_w), float(clipped_h)],
                        "area": float(clipped_w * clipped_h),
                        "iscrowd": 0,
                        "truncation": truncation,
                        "occlusion": occlusion,
                        "original_category_id": original_category,
                    }
                )
                ann_id += 1

    coco = {
        "info": {
            "description": f"VisDrone2019-DET {split_name} converted to COCO",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco, skipped


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ann_out = out_dir / "annotations"
    (out_dir / "train2017").mkdir(parents=True, exist_ok=True)
    (out_dir / "val2017").mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_name, root in [("train", args.train_root), ("val", args.val_root)]:
        coco, skipped = convert_split(root, split_name)
        year_name = f"instances_{split_name}2019.json"
        compat_name = f"instances_{split_name}2017.json"
        write_json(ann_out / year_name, coco)
        write_json(ann_out / compat_name, coco)
        summary[split_name] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "categories": len(coco["categories"]),
            "skipped": skipped,
        }

    write_json(out_dir / "conversion_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
