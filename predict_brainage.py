import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import nibabel as nib
import monai
from src.utils import get_brainage_model
from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, RandFlip, Resize, ScaleIntensity, ToTensor, SqueezeDim, AsChannelLast, NormalizeIntensity, CenterSpatialCrop


def main():

    # parse options
    # 1. Parse Options
    parser = argparse.ArgumentParser(description="Predict age from a single brain MRI scan")
    parser.add_argument("input_image", type=str, help="Path to the input NIfTI image")
    parser.add_argument("--modality", "-m", type=str, choices=["t1w", "flair"], default="t1w", help="Scan modality (determines weights used)")
    parser.add_argument("--do_preprocessing", "-p", action="store_true", help="Apply minimum preprocessingto input image (N4 bias field correction, skull-stripping, affine registration to MNI space)")
    parser.add_argument("--guided_backpropagation", "-g", action="store_true", help="Generate saliency maps for interpretability using guided backpropagation")
    args = parser.parse_args()

    # 2. Prepare Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the NIfTI file
    img_path = Path(args.input_image)
    if not img_path.exists():
        raise FileNotFoundError(f"Input file not found: {img_path}")
    
    if args.do_preprocessing:
        print("Doing Minimal Preprocessing...")
        if args.modality == "t1w":
            from src.preprocess import preprocess_t1w
            nifti_data = preprocess_t1w(img_path).numpy()
        elif args.modality == "flair":
            from src.preprocess import preprocess_flair
            nifti_data = preprocess_flair(img_path).numpy()
        else:
            raise ValueError(f"Unsupported modality: {args.modality}")    
    else:
        print("Skipping Preprocessing (Assuming input is already preprocessed)...")
        nifti_data = nib.load(str(img_path)).get_fdata()

    transforms = Compose([AddChannel(), Resize((121,145,121), mode="trilinear"), CenterSpatialCrop([113,139,119]), NormalizeIntensity(), ScaleIntensity(), ToTensor()]) 

    # Apply transforms
    # Note: Transforms usually expect inputs in (H, W, D) format
    input_tensor = transforms(nifti_data)
    
    # Add Batch Dimension (1, C, H, W, D)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 3. Load Model
    model = get_brainage_model(args.modality, device)
    model.eval()

    # 4. Run Inference
    if args.guided_backpropagation:
        from monai.visualize import gradient_based
        gbp = gradient_based.GuidedBackpropGrad(model)
        gbp_map = gbp(input_tensor).numpy()
        out_name = img_path.parent / f"{img_path.stem}_GBPmap.nii.gz"
        original_affine = nib.load(str(img_path)).affine
        nib.save(nib.Nifti1Image(gbp_map.squeeze(), original_affine), str(out_name))
        print(f"GBP map saved to: {out_name}")
        
    # Standard Inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_age = output.item()  

    # 5. Output Result
    print("-" * 30)
    print(f"Input Image: {img_path.name}")
    print(f"Predicted Brain Age: {predicted_age:.2f} years")
    print("-" * 30)
    
if __name__ == "__main__":
    main()