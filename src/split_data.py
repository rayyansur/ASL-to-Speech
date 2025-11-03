import os
import random
import shutil

from sklearn.model_selection import train_test_split

random.seed(42)

DATASET_DIR = "../data/asl_alphabet_train/asl_alphabet_train/"
DEST_DIR = "../data/asl_alphabet_split/"

train_frac = 0.8
test_frac = 0.1
val_frac = 0.1

classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
classes.sort()

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def copy_files(file_list, dst_folder):
    mkdir(dst_folder)
    for f in file_list:
        shutil.copy2(f, os.path.join(dst_folder, os.path.basename(f)))


for c in classes:
    class_folder = os.path.join(DATASET_DIR, c)
    files = [os.path.join(class_folder, f) for f in os.listdir(class_folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()
    if len(files) == 0:
        print(f"Warning: no images for class {c}")
        continue

    # First split train vs temp (val+test)
    train_files, temp_files = train_test_split(files, train_size=train_frac, random_state=42, shuffle=True,
                                               stratify=None)

    # Now split temp into val and test with relative proportions
    if (val_frac + test_frac) > 0:
        rel_val = val_frac / (val_frac + test_frac)
        val_files, test_files = train_test_split(temp_files, train_size=rel_val, random_state=42, shuffle=True,
                                                 stratify=None)
    else:
        val_files, test_files = [], []

    # destination folders per class
    train_dst = os.path.join(DEST_DIR, "train", c)
    val_dst = os.path.join(DEST_DIR, "val", c)
    test_dst = os.path.join(DEST_DIR, "test", c)

    copy_files(train_files, train_dst)
    copy_files(val_files, val_dst)
    copy_files(test_files, test_dst)

    print(f"{c}: total={len(files)} -> train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("Done. New dataset root:", DEST_DIR)