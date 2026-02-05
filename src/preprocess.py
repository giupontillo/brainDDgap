#!/usr/bin/env python3

"""
Minimum preprocessing of brain MRI images
=============================================================================================
"""

# import all relevant modules
from pathlib import Path
import ants
import antspynet

def preprocess_t1w(img_path):
    image = ants.image_read(img_path)
    image_preprocessing = antspynet.preprocess_brain_image(image, 
                                                truncate_intensity=(0.01, 0.99), 
                                                brain_extraction_modality="t1",
                                                template="croppedMni152",
                                                template_transform_type="antsRegistrationSyNQuickRepro[a]",
                                                do_bias_correction=True,
                                                do_denoising=True,
                                                verbose=True)
    preprocessed_image = image_preprocessing["preprocessed_image"]
    preprocessed_image_brain = image_preprocessing["preprocessed_image"] * image_preprocessing['brain_mask']
    return preprocessed_image_brain

def preprocess_flair(img_path):
    # use FLAIR-specific template (https://brainder.org/download/flair/)
    template_file = Path(__name__).resolve().parent / "template" / "GG-853-FLAIR-1.0mm.nii.gz"
    template = ants.image_read(template_file)
    image = ants.image_read(img_path)
    image_preprocessing = antspynet.preprocess_brain_image(image, 
                                                truncate_intensity=(0.01, 0.99), 
                                                brain_extraction_modality="flair",
                                                template=template,
                                                template_transform_type="antsRegistrationSyNQuickRepro[a]",
                                                do_bias_correction=True,
                                                do_denoising=True,
                                                verbose=True)
    preprocessed_image = image_preprocessing["preprocessed_image"]
    preprocessed_image_brain = image_preprocessing["preprocessed_image"] * image_preprocessing['brain_mask']
    return preprocessed_image_brain