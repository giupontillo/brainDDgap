#!/usr/bin/env python3

"""
Utils
=============================================================================================
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import monai
import urllib.request


# Define paths relative to this script
WEIGHTS_DIR = Path(__name__).resolve().parent.parent / "weights"

def get_brainage_model(modality, device):
    """Initializes the model and loads weights based on modality."""
    # Define model architecture
    base_model = monai.networks.nets.densenet.densenet264(spatial_dims=3, in_channels=1, out_channels=1)
    model = torch.nn.Sequential(base_model, torch.nn.Linear(1, 1)).to(device)

    # Select weights file based on modality
    if modality.lower() == "t1w":
        weights_path = WEIGHTS_DIR / "best_model_age_t1w.pth"
    elif modality.lower() == "flair":
        weights_path = WEIGHTS_DIR / "best_model_age_flair.pth"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    if not weights_path.exists():
        url = "https://zenodo.org/records/18483714/best_model_age_" + modality.lower() + ".pth?download=1"
        urllib.request.urlretrieve(url, weights_path)

    print(f"Loading {modality} weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model

def get_brainDD_model(modality, device):
    """Initializes the model and loads weights based on modality."""
    # Define model architecture
    base_model = monai.networks.nets.densenet.densenet264(spatial_dims=3, in_channels=1, out_channels=1)
    model = torch.nn.Sequential(base_model, torch.nn.Linear(1, 1)).to(device)

    # Select weights file based on modality
    if modality.lower() == "t1w":
        weights_path = WEIGHTS_DIR / "best_model_DD_t1w.pth"
    elif modality.lower() == "flair":
        weights_path = WEIGHTS_DIR / "best_model_DD_flair.pth"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    if not weights_path.exists():
        url = "https://zenodo.org/records/18483714/best_model_DD_" + modality.lower() + ".pth?download=1"
        urllib.request.urlretrieve(url, weights_path)

    print(f"Loading {modality} weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model