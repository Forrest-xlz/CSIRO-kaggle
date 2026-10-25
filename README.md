# CSIRO Image2Biomass Prediction

My final solution for the Kaggle CSIRO Image2Biomass Prediction competition.

**TL;DR.** A frozen DINOv3 ViT-7B extracts image features at three scales and three layer taps. Per-view Ridge models predict the biomass targets, and a second Ridge layer stacks their out-of-fold predictions. The final baseline blends raw-target and log-target pipelines, averages 2-, 3-, and 5-fold configurations, and applies biomass-consistency post-processing.

---

### Environment setup

The notebooks were developed for Kaggle and a local Conda environment. Use Python 3.11 or 3.12. The 7B feature extraction step requires the DINOv3 checkpoint and benefits from a CUDA-enabled NVIDIA PyTorch installation.

On Windows, activate the existing environment, install a PyTorch build that matches your GPU and driver from the official PyTorch selector at https://pytorch.org/get-started/locally/, then install the project dependencies:

    conda activate csiro
    python -m pip install -U pip setuptools wheel
    python -m pip install -r requirements.txt

Check CUDA availability:

    python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

The baseline and ablation training notebooks use cached features and do not need to load the 7B model. AutoGluon is used by the metadata-only EDA notebook.

---

### Data and model files

Place the competition files in the following local layout:

    data/
      train.csv
      test.csv
      train/
      test/

The data directory is excluded from Git. The DINOv3 checkpoint is also not included in this repository. The feature-extraction notebook currently expects these Kaggle mounts:

- Competition data: /kaggle/input/csiro-biomass
- DINOv3 checkpoint: /kaggle/input/csiro-dinov3-patch16-7b/vit_7b_patch16_dinov3.lvd1689m

For a local run, update DATA_PATH and MODEL_DIR in the extraction notebook to point to the local data and checkpoint directories.

---

### Generate reusable image features

Run features/01_extract_dinov3_7b.ipynb once. DINOv3 is used as a frozen feature extractor; it is not fine-tuned. The notebook writes dinov3_features.npz with:

- Scales 256, 512, and 768.
- The final layer and intermediate blocks 35 and 38.
- Average, max, min, and signed-GeM pooled features for each scale and layer.
- Image IDs, target values, and feature extraction metadata.

The archive contains raw pooled features before RobustScaler and PCA. It is reused by the baseline and all ablation notebooks, so each experiment does not need to run DINOv3 again. The generated NPZ file is ignored by Git.

---

### Train the baseline

Open baseline.ipynb after attaching the feature archive. In Kaggle, attach the extraction output as a dataset; the notebook searches Kaggle inputs for dinov3_features.npz. For a local run, set CFG.FEATURE_FILE to the archive path if automatic discovery does not find it.

The training pipeline:

1. Excludes image ID ID230058600, matching the submitted baseline.
2. Applies RobustScaler independently to each feature view, fitted on train and test features.
3. Fits PCA independently per view, using up to 355 components fitted on training features.
4. Trains Ridge base models with alpha 5 and Ridge meta-models with alpha 1.
5. Averages raw-space and log1p-space pipeline predictions.
6. Averages the 2-, 3-, and 5-fold outer configurations.
7. Clips negative predictions and reconciles the biomass target identities.

The notebook writes submission.csv to the current working directory or Kaggle working directory. Its printed training reference score is optimistic because the Ridge meta-model is fitted on the same out-of-fold rows used to report that reference value.

---

### Ablations

The ablations directory contains ten independent training notebooks: the baseline plus nine variants. Each notebook records its experiment change and submitted public/private score in the opening cell. All experiments use the same dinov3_features.npz archive.

The variants test image scales, intermediate layers, outer fold count, post-processing, target space, and pooling strategy. See ablations/README.md for the run order and details.

---

### EDA

The eda directory contains the original exploratory notebooks and a metadata-only AutoGluon experiment:

- eda.ipynb
- eda_metric_with_states.ipynb
- eda_noisy_data.ipynb
- eda_metadata_autogluon_5fold.ipynb

The AutoGluon notebook uses Sampling_Date, State, Species, Pre_GSHH_NDVI, and Height_Ave_cm only. It aggregates targets to one row per image, creates five-fold out-of-fold predictions for each target, and reports the competition weighted R2. It excludes ID230058600 to match the baseline.

---

### Results

Submitted baseline scores:

- Public score: 0.75696
- Private score: 0.64833

The full Kaggle write-up, including the ablation score table and discussion, is in [kaggle_writeup.md](kaggle_writeup.md).

---

### Repository contents

- baseline.ipynb: cached-feature training notebook for the final baseline.
- features/01_extract_dinov3_7b.ipynb: one-time frozen DINOv3 feature extraction.
- ablations/: baseline and ablation notebooks.
- eda/: exploratory analysis and metadata-only AutoGluon cross-validation.
- data/: local competition data, excluded from Git.
- requirements.txt: Python dependencies.
