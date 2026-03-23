import pandas as pd
import numpy as np
import random

def rotate_points(xs, ys, angle):
    cx, cy = np.mean(xs), np.mean(ys)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xs_rot, ys_rot = [], []
    for x, y in zip(xs, ys):
        x0, y0 = x - cx, y - cy
        xr = x0 * cos_a - y0 * sin_a
        yr = x0 * sin_a + y0 * cos_a
        xs_rot.append(xr + cx)
        ys_rot.append(yr + cy)
    return np.array(xs_rot), np.array(ys_rot)

def augment_landmarks(row, label):
    landmarks = row.copy().reshape(-1, 2)
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    flipped = False

    # Horizontal flip
    if random.random() < 0.5:
        xs = 1 - xs
        flipped = True

    # Rotation ±10°
    if random.random() < 0.5:
        angle = np.deg2rad(random.uniform(-10, 10))
        xs, ys = rotate_points(xs, ys, angle)

    # Translation ±0.02
    if random.random() < 0.5:
        dx, dy = np.random.uniform(-0.02, 0.02, 2)
        xs = np.clip(xs + dx, 0, 1)
        ys = np.clip(ys + dy, 0, 1)

    # Scaling 0.9–1.1
    if random.random() < 0.5:
        factor = np.random.uniform(0.9, 1.1)
        cx, cy = np.mean(xs), np.mean(ys)
        xs = np.clip((xs - cx) * factor + cx, 0, 1)
        ys = np.clip((ys - cy) * factor + cy, 0, 1)

    # Tiny noise
    if random.random() < 0.5:
        xs += np.random.normal(0, 0.005, size=xs.shape)
        ys += np.random.normal(0, 0.005, size=ys.shape)

    # 🔄 Adjust label if flipped
    if flipped:
        if label == "thumb right":
            label = "thumb left"
        elif label == "thumb left":
            label = "thumb right"

    return np.stack([xs, ys], axis=1).reshape(-1), label

# ======== Main ========
csv_path = "gestures_control.csv"  # original CSV
df = pd.read_csv(csv_path, header=None)

features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

augmented_rows = []
augmented_labels = []

for row, label in zip(features, labels):
    aug = augment_landmarks(row, label)
    augmented_rows.append(aug)
    augmented_labels.append(label)

# Save new augmented file (original CSV untouched)
df_aug = pd.DataFrame(augmented_rows)
df_aug['label'] = augmented_labels

df_aug.to_csv("gestures_control_processed1.csv", header=False, index=False)

print(f"Augmented file created: gestures_augmented.csv | Rows: {len(df_aug)}")
