import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite COCO file_name fields to portable basenames.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    return parser.parse_args()


def basename(path):
    return str(path).replace("\\", "/").rstrip("/").split("/")[-1]


def main():
    args = parse_args()
    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f)
    for image in data.get("images", []):
        image["file_name"] = basename(image["file_name"])

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
