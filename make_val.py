import os
import random
import shutil

train_dir = "dataset/train"
val_dir = "dataset/val"

os.makedirs(val_dir, exist_ok=True)

for cls in os.listdir(train_dir):
    src = os.path.join(train_dir, cls)
    dst = os.path.join(val_dir, cls)

    os.makedirs(dst, exist_ok=True)

    images = os.listdir(src)
    random.shuffle(images)
    split = int(len(images) * 0.2)

    # ย้ายรูป 20% ไปที่ val
    for img in images[:split]:
        shutil.move(
            os.path.join(src, img),
            os.path.join(dst, img)
        )