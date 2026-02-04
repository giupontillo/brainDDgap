#!/usr/bin/env python3

"""
Minimum preprocessing of 3D-FLAIR images
=============================================================================================
"""

# import all relevant modules
import os
from os.path import join as opj
from os.path import basename
from pathlib import Path
import argparse
import ants
import antspynet

def main():

    # parse options
    parser = argparse.ArgumentParser(description="Minimum preprocessing of 3D-FLAIR images")
    parser.add_argument ( "FLAIR_file", help="absolute path of raw FLAIR image")
    parser.add_argument ( "-o", "--output", help="output directory")
    args = parser.parse_args()

    # Define variables
    raw_filename = args.FLAIR_file
    output_dir = args.output
    # use FLAIR-specific template (https://brainder.org/download/flair/)
    template_file = Path(__name__).resolve().parent / "template" / "GG-853-FLAIR-1.0mm.nii.gz"
    template = ants.image_read(template_file)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    print ("\nstarting preprocessing for %s\n" % args.FLAIR_file)

    image = ants.image_read(raw_filename)
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
    # preprocessed_image_filename = opj(output_dir, basename(raw_filename).replace ("FLAIR.nii.gz","desc-preproc_FLAIR.nii.gz"));
    preprocessed_image_brain_filename = opj(output_dir, basename(raw_filename).replace ("FLAIR.nii.gz","desc-brain_FLAIR.nii.gz"));

    # write preprocessed images
    # ants.image_write(preprocessed_image, preprocessed_image_filename)
    ants.image_write(preprocessed_image_brain, preprocessed_image_brain_filename)

# if nothing else has been done yet, call main()    
if __name__ == '__main__': 
    main()