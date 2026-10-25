Thanks to the competition organizers for hosting this challenging competition, and to @zjayzz for the [public notebook](https://www.kaggle.com/code/zjayzz/1-6-vesion2), which provided helpful insights and a solid starting point for my solution.

This is the first time I have achieved such a strong result on Kaggle. I learned a lot from this competition, and below I share my approach, the experiments behind it, and the lessons I took away.

---

# Overview

With only 357 labeled images in the provided training set, I focused on limiting the number of parameters learned from the competition data. My solution uses a **frozen DINOv3 backbone for feature extraction**, followed by **lightweight, L2-regularized Ridge regressors**. This allows the supervised models to benefit from a powerful pretrained representation without fine-tuning the backbone. Most of the computational work is in feature extraction; once the features are available, fitting and experimenting with the downstream regressors is inexpensive.

The main idea is to combine complementary information across **image scales, network layers, and pooling operations**. These representations are fused through two-level Ridge stacking, with additional ensembling across raw/log target spaces and different fold counts. DINOv3’s strong frozen features and its reported differences in feature quality across layers motivated this direction; the specific choices in my pipeline were evaluated through the experiments reported below.

---

# Competition Understanding

## How I Used Leaderboard Feedback to Refine My Understanding

My understanding of the problem came from three sources: the competition description and training data, my prior knowledge, and feedback from experiments and the public leaderboard. Each source was useful, but each also had limitations. I therefore treated my initial understanding as a working hypothesis rather than a fixed conclusion.

I often thought of the process as exploring a dark space with a torch. My intuition suggested an initial direction, while each experiment illuminated another part of the landscape. Testing my own ideas and reproducing public approaches gradually helped me understand which assumptions were useful for this particular task.

This process guided questions such as whether to fine-tune the backbone or keep it frozen, how much local visual information mattered for biomass prediction, and which feature combinations were complementary. Public leaderboard feedback provided an additional, noisy signal rather than a complete picture of performance on unseen data. I used it to refine my hypotheses, while recognizing that a public-score improvement might not carry over to the private leaderboard.

## Why I Used Frozen Features with Ridge Stacking

### Small Supervised Models on Pretrained Features

![Small-data modeling](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F26182217%2F8d0b07d9a09535862c3076471cb715b4%2FOIP.jpg?generation=1774076720646935&alt=media)

With such a small labeled dataset, I wanted to make good use of pretrained visual knowledge while keeping the supervised part of the pipeline simple. Freezing the encoder allowed me to train regularized regressors on its features instead of updating the backbone on the competition labels.

I initially explored neural-network-based approaches, then shifted my focus to frozen features with classical regressors. I experimented with Ridge, XGBoost, and LightGBM as downstream models. My final baseline uses **Ridge at both stacking levels**.

### Faster Experiments with Less Tuning

![Experiment iteration](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F26182217%2F7c1e8b1a72cc0e9fa9828955d3da7fa4%2Fexperiments.png?generation=1774076747455241&alt=media)

One lesson I took from [Chris Deotte’s write-up on XGBoost tips and tricks](https://www.kaggle.com/writeups/cdeotte/xgboost-tips-and-tricks) was the value of making experimentation efficient.

In my workflow, Ridge models were quick to fit and required relatively little tuning compared with the neural-network training setups I had explored. This let me spend more time testing higher-level choices—such as backbone size, image resolution, layer selection, pooling, and ensembling—instead of repeatedly tuning learning rates, batch sizes, and training schedules.

### Reusing the Pipeline with Stronger Backbones

I explored progressively larger backbones, from Base to Large, Huge, and 7B, and generally observed better performance with stronger features in my experiments.

Each backbone change still required extracting new features and refitting the regressors. However, I could reuse the same downstream modeling approach with little redesign or retuning. This made it straightforward to test whether a stronger visual representation would improve the solution.

---

# Model

![Frozen DINOv3 with multi-view Ridge stacking](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F26182217%2F6357d923e744893b2b085ea14967861b%2Fcsiro_baseline_architecture.png?generation=1792936322686244&alt=media)

The core idea is to combine complementary visual features across image scales and network depths, then fuse their predictions using lightweight Ridge stacking.

Each image is split into left and right halves, with each half resized to **256 × 256, 512 × 512, and 768 × 768**. A frozen **DINOv3 ViT-7B** extracts features from **three layer taps**: intermediate blocks 35 and 38, and the final output. This produces **3 scales × 3 layers = 9 feature views**. Within each view, left and right patch tokens are concatenated along the token dimension before applying mean, max, min, and signed-GeM pooling. The four pooled vectors are concatenated, then processed with RobustScaler and PCA.

For each view, a first-level Ridge model predicts the five biomass targets. A separate second-level Ridge model for each target combines the nine corresponding base predictions, using base-level out-of-fold predictions for training. At inference time, the base predictions are averaged across folds before being passed to the second-level models.

I train separate stacks on **raw targets** and **log1p-transformed targets**, then blend their predictions equally after mapping the log-space outputs back with expm1. Finally, I average predictions from **2-, 3-, and 5-fold** configurations. Post-processing applies a linear constraint projection, clips negative values, and recomputes **GDM = Green + Clover** and **Total = GDM + Dead**.

## Implementation Details

- **Layer taps:** block indices 35 and 38 are zero-based. The backbone’s final normalization is also applied to the intermediate features, and CLS/prefix tokens are excluded from pooling.
- **Pooling and dimensionality:** signed-GeM uses p = 3. Concatenating the four pooling outputs gives a 4C-dimensional embedding per view, where C is the token feature dimension. PCA retains up to 355 components.
- **Ridge settings:** the first-level models use α = 5, and the second-level models use α = 1. All views within a stack use the same shuffled KFold splits with seed 42.
- **Feature preprocessing:** in the supplied notebook, RobustScaler is fitted on combined training and test features, while PCA is fitted on the full training features. Both are fitted once per view before the KFold loop.
- **Data filtering:** the baseline notebook excludes image `ID230058600` before feature extraction.

## Key Points

- **Multiple image scales:** combine features extracted at 256, 512, and 768 pixels per half-image.
- **Multiple feature depths:** combine the final output with intermediate blocks 35 and 38.
- **Multiple pooling operations:** concatenate mean, max, min, and signed-GeM features.
- **Lightweight supervised fusion:** use L2-regularized Ridge regression at both stacking levels.
- **Prediction ensembling:** blend raw/log-space predictions and average stacks trained with different fold counts.

---

# Ablation Study

The table below compares the full baseline with the tested variants.

| Variant | Public score | Private score |
| --- | ---: | ---: |
| Full model | **0.75696** | 0.64833 |
| Single scale: 768 | 0.75332 | 0.64324 |
| Scales: 256 + 512 | 0.74585 | 0.63738 |
| Without intermediate layers | 0.74969 | 0.62957 |
| Intermediate tap: block 35 only | 0.75466 | 0.64466 |
| Intermediate tap: block 38 only | 0.75218 | 0.62950 |
| 5-fold only | 0.75493 | 0.64774 |
| Without post-processing | 0.75208 | **0.64986** |
| Raw-space only | 0.74358 | 0.64147 |
| Average pooling only | 0.72056 | 0.60662 |

The main observations were:

- **Pooling had the largest impact among the tested variants.** Replacing multi-pooling with average pooling reduced the public score by 0.03640 and the private score by 0.04171.
- **Intermediate-layer features were also valuable.** Removing them reduced the private score by 0.01876. Using both intermediate taps performed better than either single-tap variant in these experiments.
- **Multi-scale features and raw/log-space blending improved both reported scores.** The full model outperformed the tested reduced-scale and raw-only configurations.
- **Ensembling fold counts gave a smaller gain.** Compared with using five folds alone, the private-score improvement was only 0.00059.
- **Post-processing showed a trade-off.** It improved the public score but reduced the private score by 0.00153. In fact, the variant without post-processing achieved the highest private score in this table.

These results made feature extraction and aggregation the most promising areas for further exploration in this pipeline.

---

# What Didn’t Work

- **Stacked XGBoost:** I tried a simple XGBoost-based stacking setup, but it did not improve performance in my experiments. I therefore kept Ridge as the regressor at both stacking levels.

---

# Future Work

## Feature Aggregation and Layer Selection

The largest performance drops in the reported ablations came from simplifying the pooling strategy and removing intermediate-layer features. I would therefore continue testing alternative layer combinations, pooling strategies, and small attention-based aggregation modules.

## Learning from EDA

During EDA, I observed that sampling dates were unevenly distributed across States, making geographic and seasonal effects difficult to separate in the available data. Target distributions also differed substantially across States: in the examined training subset, NSW samples had zero Dry_Clover_g, while WA samples had zero Dry_Dead_g.

These observations were not incorporated into the final solution, but they suggest useful diagnostics for understanding which dataset patterns an image-only model has learned.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F26182217%2F6c1a8200e24651b747eeccee3c1be82e%2Fstate1.png?generation=1792944132609983&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F26182217%2Fa28fb5f16043ad12453113d9698e21e9%2Fstate2.png?generation=1792944143471873&alt=media)

EDA revealed clear relationships between State, sampling date, and biomass targets. These observations could help us check whether the model has learned important patterns in the data. Where it falls short, EDA-derived knowledge may provide useful guidance for improving predictions.

## Learning from Training-Only Metadata

The training data include additional information such as species, NDVI, and height. Although these fields are unavailable at test time, they could still provide useful supervision during training—for example, through auxiliary tasks in a multi-task learning setup. This may help the model learn better visual features and improve generalization when labeled images are limited.

---

# References

- [DINOv3](https://arxiv.org/abs/2508.10104), including Appendix B.2 on per-layer feature analysis.
- [Stacked Generalization: An Introduction to Super Learning](https://www.researchgate.net/publication/345718900_Stacked_Generalization_An_Introduction_to_Super_Learning)
- [Stacked Generalization: When Does It Work?](https://www.researchgate.net/publication/2605515_Stacked_Generalization_when_does_it_work)
- [GeM pooling](https://arxiv.org/abs/1711.02512)
- [XGBoost Tips and Tricks — Chris Deotte](https://www.kaggle.com/writeups/cdeotte/xgboost-tips-and-tricks)
- [Multitask Learning — Rich Caruana (1997)](https://www.cs.cornell.edu/~caruana/mlj97.pdf)
- [A Generalist Agent (Gato) — Scott Reed et al., DeepMind (2022)](https://arxiv.org/abs/2205.06175)