import os
from glob import glob

# Paths for segmentation
base_dir = "dataset/Original images/"
img_paths = glob(os.path.join(base_dir, "Microscope 1/images/*.png")) + \
            glob(os.path.join(base_dir, "Microscope 2/images/*.png"))
mask_paths = glob(os.path.join(base_dir, "Microscope 1/masks/*.png")) + \
             glob(os.path.join(base_dir, "Microscope 2/masks/*.png"))

# Make sure order matches
img_paths = sorted(img_paths)
mask_paths = sorted(mask_paths)

print("Total images:", len(img_paths))
print("Total masks:", len(mask_paths))
