import csv
import random
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.utils import save_image  # writes tensors as PNGs

# ---------- CONFIG --------------------------------------------------------------------
SPLIT = "test"         # "train" or "test"
NUM_PER_CLASS = 10     # how many images per class
OUT_DIR = Path("example_dataset")  # folder to create / reuse
# --------------------------------------------------------------------------------------

# 1. Dataset ----------------------------------------------------------------------------
transform = transforms.ToTensor()
dataset = datasets.CIFAR10(
    root="data", train=(SPLIT == "train"), download=True, transform=transform
)

# 2. Collect indices by class ----------------------------------------------------------
class_to_indices = {i: [] for i in range(10)}

for idx in range(len(dataset)):
    _, label = dataset[idx]
    class_to_indices[label].append(idx)

# 3. Select 10 random images per class -------------------------------------------------
selected_indices = []
for label in class_to_indices:
    selected = random.sample(class_to_indices[label], NUM_PER_CLASS)
    selected_indices.extend(selected)

# 4. Output setup -----------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
csv_file = OUT_DIR / "labels.csv"

# 5. Save images and write CSV ---------------------------------------------------------
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "objects"])

    for i, idx in enumerate(selected_indices):
        img_tensor, label_id = dataset[idx]
        class_name = dataset.classes[label_id]
        filename = f"{class_name}_{i:03d}.jpg"

        save_image(img_tensor, OUT_DIR / filename)
        writer.writerow([filename, class_name])

print(f"\n✓ Saved {NUM_PER_CLASS * 10} images to {OUT_DIR}")
print(f"✓ Labels CSV written to {csv_file}")
