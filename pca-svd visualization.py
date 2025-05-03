import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from glob import glob
import random

from config import *

# --- User parameters: point these at your downloaded VMMRDB ---
data_dir = IMAGES_PATH  # Base directory containing class subdirectories
CLASSES = ["ford_escape", "ford_explorer", "honda_civic", "honda_odyssey",
           "mitsubishi_lancer", "mitsubishi_outlander", "nissan_altima", "nissan_maxima"]  
N_PER_CLASS = 50   # Number of images per class to load

# --- 1) Load & flatten images ---
def load_flattened_images(data_dir, classes, n_per_class, img_size=(64,64)):
    X, y = [], []
    for cls in classes:
        # Use glob to find all .jpg files in the class directory
        files = glob(os.path.join(data_dir, cls, "*.jpg"))[:n_per_class]
        for f in files:
            img = Image.open(f).resize(img_size).convert("RGB")
            X.append(np.asarray(img).ravel())
            y.append(cls)
    return np.vstack(X), np.array(y)

X, y = load_flattened_images(data_dir, CLASSES, N_PER_CLASS)

# --- 2) Standardize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3) PCA & SVD ---
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)

svd = TruncatedSVD(n_components=2, random_state=0)
X_svd = svd.fit_transform(X_scaled)

# --- 4) Plotting utility ---
def scatter_proj(X_proj, title, ax):
    for cls in np.unique(y):
        mask = (y == cls)
        ax.scatter(X_proj[mask,0], X_proj[mask,1],
                   label=cls, alpha=0.7, edgecolor='k', s=50)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")

# --- 5) Make the figure ---
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))

scatter_proj(X_pca, "VMMRDB – PCA (1st 2 Components)", ax1)
scatter_proj(X_svd, "VMMRDB – SVD (1st 2 Components)", ax2)

for ax in (ax1, ax2):
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

plt.tight_layout()
plt.show()