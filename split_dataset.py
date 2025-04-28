import os
import random
import shutil

def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=None):
    """Splits an image dataset into training, validation, and testing sets, preserving the subfolder structure.

    Args:
        input_dir (str): Path to the directory containing the original dataset with subfolders.
        output_dir (str): Path to the directory where the split dataset will be created.
        train_ratio (float): Ratio of images to use for training (default: 0.8).
        val_ratio (float): Ratio of images to use for validation (default: 0.1).
        test_ratio (float): Ratio of images to use for testing (default: 0.1).
        random_seed (int, optional): Seed for random shuffling (for reproducibility).
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not (train_ratio + val_ratio + test_ratio == 1):
        raise ValueError("Train, validation, and test ratios must sum to 1")

    if random_seed is not None:
        random.seed(random_seed)

    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        images = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        num_images = len(images)
        train_split = int(train_ratio * num_images)
        val_split = int(val_ratio * num_images)

        train_images = images[:train_split]
        val_images = images[train_split:train_split + val_split]
        test_images = images[train_split + val_split:]

        _move_images(train_images, os.path.join(output_dir, "train", subfolder_name))
        _move_images(val_images, os.path.join(output_dir, "val", subfolder_name))
        _move_images(test_images, os.path.join(output_dir, "test", subfolder_name))

def _move_images(image_paths, destination_dir):
    """Moves image files to the specified directory, creating it if necessary."""
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for image_path in image_paths:
        shutil.copy(image_path, destination_dir)

if __name__ == '__main__':
    input_directory = "./dataset" # Replace with the actual path to your dataset
    output_directory = "./split_dataset" # Replace with the desired output path
    split_dataset(input_directory, output_directory, random_seed=42)