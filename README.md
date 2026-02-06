# The Brain-Predicted DD and MS-age Gaps 🧠
## About 💡
**Disease Duration gap**
![Main figure](images/Figure_1.svg)

**Multiple Sclerosis-age gap**
![MS-age gap](images/SupplementaryFigure_1.svg)

## Installation 📦
### 1. Clone the Repository
```
git clone https://github.com/giupontillo/brainDDgap.git
cd brainDDgap
```
### 2. Set Up Environment
We recommend using Conda to manage dependencies. This will install the package in editable mode (-e .), allowing you to modify the source code without re-installing.
```
conda env create -f environment.yml
conda activate brainDDgap
```
### 3. Download Model Weights
Model weights are not stored in this repository due to size. Download model weights from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18483714.svg)](https://doi.org/10.5281/zenodo.18483714) and put them in the weights folder (this is done automatically if weights are not found).

## Usage 🛠️
### Command Line Interface
Once installed, you can run the prediction directly from your terminal using the predict_brainage and predict_brainDD commands.
```
# Basic prediction
predict_brainage path/to/scan.nii.gz --modality t1w --do_preprocessing

# Prediction with GBP Map generation
predict_brainDD path/to/scan.nii.gz --modality flair --do_preprocessing --guided_backpropagation
```

**Arguments**
- input_image: Path to the NIfTI (.nii.gz) file.
- `--modality`: Choose between t1w or flair.
- `--do_preprocessing`: do preprocessing.
- `--guided_backpropagation`: generate GBP map showing predictive regions.
- `--output`: output folder where GMP map and preprocessed image are saved (only works with --guided_backpropagation, default folder is the same as the input image)

### Run directly in Python
```
from brainDDgap.preprocess import preprocess_t1w
```

## Project Structure 📂 
```
├── src/
│   ├── preprocess.py       # Unified T1/FLAIR preprocessing logic
│   ├── utils.py            # Model initialization and weight loading
│   └── template/           # Reference templates for registration          
├── weights/                # Pre-trained model weights (to download)
├── predict_brainage.py     # Main entry point script
├── predict_brainDD.py      # Main entry point script
├── environment.yml         # Conda environment definition
└── pyproject.toml          # Python package configuration
```

## Citation 📝
If you use this code or the pre-trained models in your research, please cite:

https://www.neurology.org/doi/10.1212/WNL.0000000000209976

https://doi.org/10.5281/zenodo.18483714

