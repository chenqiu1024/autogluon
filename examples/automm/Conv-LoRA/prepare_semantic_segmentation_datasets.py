import argparse
import os

from autogluon.common.loaders import load_zip


def get_data_home_dir():
    return os.path.join(os.getcwd().replace("\\", "/"), "datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[
            "polyp",
            "leaf_disease_segmentation",
            "camo_sem_seg",
            "isic2017",
            "road_segmentation",
            "SBU-shadow",
        ],
        help="Which semantic segmentation datasets to download/unzip.",
    )
    args = parser.parse_args()

    base_dir = get_data_home_dir()
    for name in args.datasets:
        url = f"s3://automl-mm-bench/semantic_segmentation/{name}.zip"

        dataset_dir = os.path.join(base_dir, name)
        load_zip.unzip(url, unzip_dir=dataset_dir)
