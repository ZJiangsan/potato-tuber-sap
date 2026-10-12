# Potato tuber disease: dual-head CNN-Transformer with segmentation-aware pooling (SAP)

Analysis code for the paper *[Paper title]* (Postharvest Biology and Technology, [year], DOI: [doi]).

One network classifies the disease of a potato tuber and segments its lesions. A ResNet-34 encoder and a Transformer bottleneck feed two heads:

- a decoder with U-Net skip connections that predicts the lesion mask;
- a classification head that pools the bottleneck features.

The hybrid baseline pools with global average pooling (GAP). **Segmentation-aware pooling (SAP)** pools the features weighted by the predicted mask, with a stop-gradient on the mask:

```
v = Σ F · M̃ / (Σ M̃ + ε)      M̃ = stopgrad(sigmoid(seg_logits))
```

## Evaluation protocol (no data leakage)

- **Split before augmenting.** Folds are built from the original images. Augmentation (Albumentations) is applied on the fly to training images only. Every test image is an unseen original.
- **Grouped, stratified 5-fold cross-validation** (`StratifiedGroupKFold`). Two kinds of images always land in the same fold:
  - near-duplicates (dHash Hamming distance ≤ 6);
  - camera photos of the same class taken within 15 s, which are treated as the same tuber.
- **Inner validation split.** 1/8 of each training fold is held out for checkpoint selection and early stopping. The test fold is never used for these choices.
- **Checkpoint selection.** Models are selected on validation Dice. The revised SAP models are selected on validation total loss and input-level gating on validation cross-entropy.
- **Scheduler and stopping.** ReduceLROnPlateau runs on validation loss. Training stops early after 40 epochs without improvement, with at most 150 epochs. SAP models are fine-tuned from the hybrid for at most 50 epochs.

## Models compared

| key | model |
|---|---|
| `hybrid` | Hybrid CNN-Transformer, GAP head (baseline) |
| `hybrid_sapd` | Hybrid with SAP, stop-gradient on the mask (**proposed**) |
| `hybrid_sap` | SAP without stop-gradient |
| `hybrid_msap` | Multi-scale SAP (1/8, 1/16 and the bottleneck) |
| `hybrid_early` | Input-level gating: the frozen hybrid segments, then a second ResNet-34 classifies only the predicted lesion pixels |
| `cnn_only` | ResNet-34 encoder–decoder without the Transformer or skip connections |
| `unet_r34` | U-Net with a ResNet-34 encoder |
| `att_unet` | Attention U-Net with a ResNet-34 encoder |
| `trans_only` | Transformer-only |
| `vit_pre` | ViT-B/16, ImageNet-pretrained |

## Data

The images are not included in this repository ([data availability statement / link]). The script expects this layout:

```
potato_diseaseN/
  Black_Scurf/   1.jpg 1.json  2.jpg 2.json ...
  Common_Scab/   ...
  DR_I/  DR_S/  Dry_Rot/  Pink_Rot/  Powdery_Scab/  Silver_Scurf/
```

Each `.json` file is an AnyLabeling / LabelMe annotation containing lesion polygons. Folder names map to class names in `CLASS_NAMES`. Four images that were found by hand to be repeated shots are listed in `EXCLUDE`.

For the cross-source test, in-house images are identified by camera file names (`IMG_YYYYMMDD_HHMMSS`). Everything else counts as public. An optional `source.csv` in the data folder (columns `id,source`, e.g. `Black_Scurf/12,public`) overrides that rule.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # install the torch build that matches your CUDA first if needed
```

The runs reported in the paper used one NVIDIA V100 (32 GB).

## Running

`potato_tuber_analysis.py` is a flat cell script (`# %%` cells) with no `main()`. You can run it from top to bottom:

```bash
POTATO_DATA=/path/to/potato_diseaseN POTATO_OUT=/path/to/output python potato_tuber_analysis.py
```

You can also run it cell by cell in VS Code, Spyder or Jupyter. If `POTATO_DATA` and `POTATO_OUT` are not set, the script uses `~/potato_diseaseN` and `~/potato_revision_v4`.

- **Smoke test.** Set `SMOKE = True` in the config cell for a 2-epoch dry run of the whole pipeline. Its outputs go to a separate `*_smoke` folder.
- **GPU.** `GPU_ID` at the top selects the GPU by its `nvidia-smi` index.
- **Resumable.** Each finished model × seed × fold run is saved in `runs/` and `ckpt/` and skipped on re-run.
- **Settings guard.** The key settings are stored in `config.json`. If they change, the script stops rather than mixing results from different settings. Use a new output folder in that case.
- **Check the masks first.** `figures/check_masks.png` overlays the annotations on four images per class. Look at it before the long training run.

The full run (10 models, 3 seeds for the hybrid and SAP variants, 5 folds, plus the cross-source runs) takes [about N hours] on a V100.

## Outputs (in the output folder)

| file | content |
|---|---|
| `groups.csv`, `near_duplicates.csv`, `folds.csv` | grouping of linked images and fold assignment |
| `per_image_results.csv` | out-of-fold predictions for every model, seed and image |
| `table_models.csv` | accuracy, macro-F1, pixel precision / recall / Dice / IoU, lesion-area R² and MAE, bootstrap 95% CI, fold mean ± SD |
| `table_seeds.csv` | results per seed |
| `table_per_class.csv` | per-class results of the proposed model |
| `table_sap_vs_baseline.csv` | SAP vs hybrid: McNemar, Wilcoxon, Grad-CAM lesion share, occlusion tests |
| `table_sap_variants.csv`, `table_sap_variants_tests.csv` | all SAP variants vs hybrid, 3 seeds |
| `table_early_masking.csv`, `table_early_vs_hybrid.csv` | input-level gating |
| `table_background_swap.csv`, `table_background_swap_tests.csv` | lesions pasted onto the background of another class |
| `table_lime.csv` | LIME: share of positive evidence on the lesion |
| `table_uncertainty.csv` | test-time augmentation: entropy AUROC for errors, lesion-area SD vs error |
| `table_complexity.csv` | parameters, MACs (including attention), GPU and CPU latency |
| `table_cross_source*.csv` | trained on public images, tested on in-house images |
| `figures/*.png` | mask check, confusion matrix, lesion-area scatter, Grad-CAM lesion share and examples, background-swap examples, training curves |

## Citation

If you use this code, please cite:

```
[Authors] ([year]). [Paper title]. Postharvest Biology and Technology, [volume], [article]. https://doi.org/[doi]
```

## License

MIT. See `LICENSE`.

> **Upload status:** The Python entry point from the source ZIP is pending upload. Do not treat this repository as complete until `potato_tuber_analysis.py` appears in its root.
