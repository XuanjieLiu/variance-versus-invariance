import argparse
import os
import random
import shutil
from copy import deepcopy
from glob import glob

from tqdm import tqdm


def split_train_val_test(data_dir, output_dir, val_percentage=0.1, test_percentage=0.1):
    """
    data_dir: directory directly containing .png files
    output_dir: directory to store the split data
    val_percentage: percentage of data to be used as validation set
    test_percentage: percentage of data to be used as test set
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    png_paths = glob(os.path.join(data_dir, "*.png"))
    n_val = int(len(png_paths) * val_percentage)
    n_test = int(len(png_paths) * test_percentage)
    print(f"{len(png_paths)} png files found.")
    random.shuffle(png_paths)
    for i, png_path in tqdm(enumerate(png_paths)):
        if i < n_val:
            shutil.copy(png_path, val_dir)
        elif i >= n_val and i < n_test + n_val:
            shutil.copy(png_path, test_dir)
        elif i >= n_test + n_val:
            shutil.copy(png_path, train_dir)

    print(
        f"Splited into {len(glob(os.path.join(train_dir, '*.png')))} training files and {len(glob(os.path.join(val_dir, '*.png')))} validation files and {len(glob(os.path.join(test_dir, '*.png')))} test files."
    )


def split_ood_few_shot(data_dir, output_dir, n_shots=[1, 5, 10]):
    """
    files selected for few-shot learning will not be shown in the test set.
    The style label is the last word of the file name.
    """
    png_paths = glob(os.path.join(data_dir, "*.png"))
    test_png_paths = deepcopy(png_paths)
    for n_shot in n_shots:
        shots = {}
        shot_dir = os.path.join(output_dir, f"{n_shot}_shot")
        os.makedirs(shot_dir, exist_ok=True)
        for png_path in png_paths:
            png_name = os.path.basename(png_path)
            s_label = png_name.split("_")[-1].split(".")[0]
            if s_label not in shots:
                shots[s_label] = []
            if len(shots[s_label]) < n_shot:
                shots[s_label].append(png_path)
                if png_path in test_png_paths:
                    test_png_paths.remove(png_path)
        for paths in shots.values():
            for path in paths:
                shutil.copy(path, shot_dir)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    for png_path in test_png_paths:
        shutil.copy(png_path, test_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/lowercase_letters_")
    parser.add_argument("--save_dir", type=str, default="../data/LowercaseLetters")
    parser.add_argument("--val_percentage", type=float, default=0.1)
    parser.add_argument("--test_percentage", type=float, default=0.1)
    parser.add_argument("--split_for_ood", action="store_true")

    args = parser.parse_args()
    if args.split_for_ood:
        split_ood_few_shot(args.data_dir, args.save_dir, n_shots=[1, 5, 10])
    else:
        split_train_val_test(
            args.data_dir,
            args.save_dir,
            args.val_percentage,
            args.test_percentage,
        )
