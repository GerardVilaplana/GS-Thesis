# HandAffordSplat model card

## Model

The selected HandAffordSplat model is a PointNet-style per-Gaussian binary
classifier. A shared encoder maps each Gaussian feature vector to a latent
embedding. Mean pooling produces object-level context, which is concatenated
back to every local embedding before classification.

The final input has 401 values per Gaussian:

- 384 object-crop DINOv2 values.
- 14 Gaussian values: normalized position, scale, rotation, opacity, and RGB.
- 3 DINOv2 assignment-quality values: validity, weight, and visible views.

## Intended use

The model predicts whether each Gaussian in an already segmented object belongs
to a handle or graspable region. It is a perception component, not a grasp
planner. Predictions do not encode reachability, grasp pose, task constraints,
or physical feasibility.

## Training data

The model was trained on object-level Gaussian representations reconstructed
from HANDAL image sequences. Two-dimensional object and handle annotations were
lifted into the Gaussian representation. Ten low-quality reconstructions were
excluded, leaving 535 scenes from 17 categories.

## Checkpoints

- seen/model.pt: categories represented during training; threshold 0.58.
- loco/<category>/model.pt: one checkpoint per held-out category.

Each checkpoint is self-contained and stores model weights, training feature
mean and standard deviation, architecture metadata, selected validation
threshold, and evaluation metadata.

## Evaluation

| Setting | mIoU | F1 |
|---|---:|---:|
| Seen test | 0.855 | 0.917 |
| LOCO test | 0.672 | 0.763 |

Metrics are first computed per scene. Seen values average all test scenes. LOCO
values average scenes within each held-out category and then average the 17
category results equally.

## Limitations

Performance is weaker for functional geometries that differ from those in the
training categories, notably mugs and power drills in LOCO evaluation. Errors in
reconstruction or language-guided object segmentation can propagate to handle
prediction. The in-the-wild evaluation is qualitative and covers only a small
number of scenes.
