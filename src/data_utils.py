import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(img_dir, label_dir, train_ratio=0.8):
    """
    Chia dataset thành train/val.
    """
    images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
    train_files, val_files = train_test_split(images, train_size=train_ratio, random_state=42)

    for split, files in [("train", train_files), ("val", val_files)]:
        os.makedirs(f"data/images/{split}", exist_ok=True)
        os.makedirs(f"data/labels/{split}", exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(img_dir, f), f"data/images/{split}/{f}")
            label_file = f.replace(".jpg", ".txt").replace(".png", ".txt")
            if os.path.exists(os.path.join(label_dir, label_file)):
                shutil.copy(os.path.join(label_dir, label_file), f"data/labels/{split}/{label_file}")

if __name__ == "__main__":
    split_dataset("raw_images", "raw_labels", train_ratio=0.8)
