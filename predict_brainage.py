from pathlib import Path
import torch
import argparse
import nibabel as nib
import numpy as np
from monai.transforms import AddChannel, Compose, Resize, ScaleIntensity, ToTensor, NormalizeIntensity, CenterSpatialCrop
from src.utils import get_brainage_model


def main():

    # parse options
    # 1. Parse Options
    parser = argparse.ArgumentParser(description="Predict MS-age from a single brain MRI scan")
    parser.add_argument("input_image", type=str, help="Path to the input NIfTI image")
    parser.add_argument("--modality", "-m", type=str, choices=["t1w", "flair"], default="t1w", help="Scan modality (determines weights used)")
    parser.add_argument("--do_preprocessing", "-p", action="store_true", help="Apply minimum preprocessingto input image (N4 bias field correction, skull-stripping, affine registration to MNI space)")
    parser.add_argument("--guided_backpropagation", "-g", action="store_true", help="Generate saliency maps for interpretability using guided backpropagation")
    parser.add_argument("--output", "-o", type=str, default=Path(args.input_image).parent, help="Output folder where preprocessed images and GBP maps are saved (only used if GBP is performed)")
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

    # Apply transforms (Transforms expect inputs in (H, W, D) format)
    input_tensor = transforms(nifti_data)
    
    # Add Batch Dimension (1, C, H, W, D)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 3. Load Model
    model = get_brainage_model(args.modality, device)
    model.eval()

    # 4. Run Inference
    if args.guided_backpropagation:
        from monai.visualize import gradient_based
        with torch.set_grad_enabled(True):
            gbp = gradient_based.GuidedBackpropGrad(model)
            gbp_map = gbp(input_tensor).cpu().numpy()
            preproc_img = input_tensor.cpu().numpy()
            output_folder = Path(args.output)
            out_name_gbp_map = output_folder / f"{Path(img_path).stem}_MSage-GBP.nii.gz"
            out_name_preproc_img = output_folder / f"{Path(img_path).stem}_preproc.nii.gz"
            nib.save(nib.Nifti1Image(gbp_map.squeeze(), np.eye(4)), str(out_name_gbp_map))
            nib.save(nib.Nifti1Image(preproc_img.squeeze(), np.eye(4)), str(out_name_preproc_img))
            print(f"GBP map saved to: {out_name_gbp_map}")
            print(f"preprocessed image saved to: {out_name_preproc_img}")
        
    # Standard Inference
    with torch.no_grad():
        output = model(input_tensor)
        predicted_age = output.item()  

    # 5. Output Result
    print("Image,Modality,Brain-predictedMSage_years")
    print(f"{img_path},{args.modality},{predicted_age:.2f}")
    
if __name__ == "__main__":
    main()