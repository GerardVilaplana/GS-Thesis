# Thesis results

## Final model

The selected model is PointNet Mean with object-crop DINOv2, the full RGB
Gaussian feature configuration, and assignment-quality values.

| Representation | Seen mIoU | Seen F1 | LOCO mIoU | LOCO F1 |
|---|---:|---:|---:|---:|
| Supervised HANDAL | 0.855 | 0.917 | 0.672 | 0.763 |
| Language-guided | 0.831 | 0.891 | 0.653 | 0.733 |

The supervised and language-guided rows use separately reconstructed Gaussian
representations. They evaluate each complete pipeline rather than isolating only
the segmentation stage.

## Validation selection

The final feature configuration reached 0.839 seen mIoU and 0.706 LOCO mIoU on
validation data. These validation results were used for configuration selection;
the test values above provide the final evaluation.

## External baselines

On supervised HANDAL test representations, the best validation-threshold result
was 0.383 mIoU for OpenAD and 0.368 mIoU for AffordSplatNet. These pretrained
models were evaluated without retraining on HANDAL, so the comparison measures
transfer to the HANDAL handle definition.
