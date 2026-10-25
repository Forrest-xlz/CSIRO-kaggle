# Ablation notebooks

Run features/01_extract_dinov3_7b.ipynb once, save its output as a Kaggle dataset, and attach it to each training notebook. The baseline and ablation notebooks train from cached features and do not load DINOv3 or decode images again.

The archive contains the final layer and blocks 35 and 38 at scales 256, 512, and 768. Each view stores four concatenated raw vectors in this order: average, max, min, signed GeM. Each notebook selects the scales, blocks, or pooling mode required by its experiment.

Header scores are the submitted Kaggle public and private scores supplied for the study. Runtime reference scores are computed on fitted training meta-model predictions and are optimistic.