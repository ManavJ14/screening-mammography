# Screening Mammography: Uncertainty-Aware Breast Mass Classification

A PyTorch pipeline that classifies mammogram images as **benign or malignant** using the CBIS-DDSM dataset. Beyond a plain classifier, the model estimates its own **uncertainty** (Monte Carlo dropout, optional deep ensembles) and uses it to simulate a **triage workflow**: confident exams could be auto-read, uncertain ones sent to a radiologist or arbitration.

*Minor project, Roll No. 24U022028. Full write-up: [`Manav_Minor_Report_24U022028.pdf`](Manav_Minor_Report_24U022028.pdf)*

> **Research and educational use only.** This is not a medical device and must not be used for diagnosis or clinical decisions.

## Repository Structure

```
screening-mammography/
├── Screening_mammography_24u022028.ipynb   # End-to-end notebook: data, model, training, evaluation, plots
├── Manav_Minor_Report_24U022028.pdf        # Project report
└── README.md
```

## Dataset

- **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM), JPEG version on Kaggle: [`awsaf49/cbis-ddsm-breast-cancer-image-dataset`](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- **Task:** binary classification of **mass** cases, malignant (1) vs benign (0), from the full mammogram image.
- **Training set:** 1,000 images, class-balanced (500 / 500), sampled from the official CBIS-DDSM **train** split.
- **Test set:** 200 images, class-balanced (100 / 100), from the official **test** split. Patients do not overlap between train and test.
- **Auxiliary labels:** BI-RADS breast density (4 classes) and the lesion ROI mask, used as extra training signals where available.

The dataset is downloaded on demand through `kagglehub` (or read from `/kaggle/input` when running on Kaggle). Only the files needed are fetched.

## Method

| Component | Details |
|---|---|
| Backbone | EfficientNet-B0, ImageNet-pretrained (EfficientNet-B1 and MobileNetV3-Large also supported) |
| Input | Grayscale, resized to 512×512, replicated to 3 channels, normalised (mean 0.449, std 0.226) |
| Augmentation | Random horizontal flip |
| Classification head | Global average pool, dropout (p = 0.3), linear layer; binary cross-entropy |
| Auxiliary heads | Breast-density classifier (weight 0.2) and lesion-mask segmentation head (weight 0.5) |
| Uncertainty | Monte Carlo dropout (20 samples), optional deep ensemble; predictive entropy split into aleatoric and epistemic parts |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4, batch size 16, 50 epochs, seed 0 |
| Triage | Uncertainty thresholds route exams to auto-read, routine reading, or arbitration, tuned to a sensitivity floor of 95% |

## Getting Started

The notebook was developed on **Google Colab** with a GPU.

```bash
git clone https://github.com/ManavJ14/screening-mammography.git
cd screening-mammography
pip install torch torchvision numpy pandas pillow scikit-learn matplotlib kagglehub jupyter
```

**Kaggle credentials.** Add `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` as secrets (Colab: sidebar key icon, or set them as environment variables locally). **Never paste credentials into the notebook itself.**

Then open `Screening_mammography_24u022028.ipynb` and run the cells in order: data and model definitions, training (`Cfg` holds all hyperparameters), then the held-out evaluation sections. To use more test images, change `N_TEST`.


## Author

**Manav Jadon** ([@ManavJ14](https://github.com/ManavJ14))

## Acknowledgements

CBIS-DDSM: Lee et al., *A curated mammography data set for use in computer-aided detection and diagnosis research*, Scientific Data, 2017.