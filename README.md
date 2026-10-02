# Site-Specific Visibility Networks

Code and result files for the M.S. thesis *Site-Specific Adaptation of Convolutional Neural Networks
for Visibility Estimation from Aviation Weather Camera Images* (Sreelasya Mutukula, Ohio University,
2026).

The thesis trains three networks, VisNet, RMEP and VisNet-Similarity, to estimate visibility from FAA
weather camera images in ten classes from 1 to 10 statute miles. Each network is trained in three
ways: pooled over all sites, from scratch at each site, and fine-tuned at each site from the pooled
model. Everything is in the `vis_networks` folder, and all commands below are run from there.

## Reproduce the thesis tables and figures

These two scripts read only the result files in this repository, so they need no images:

```
cd vis_networks
python make_thesis_tables.py     # writes thesis_tables/ (table bodies and numbers.txt)
python make_thesis_figures.py    # writes thesis_figures/
```

`thesis_tables/numbers.txt` records every value the scripts compute, including the nine
configurations on the 979-image evaluation set and the paired bootstrap comparisons.

## Requirements

The experiments ran on an NVIDIA GeForce RTX 4080 with CUDA 11.8. `environment_freeze.txt` lists
the full environment (UTF-16 encoded). Its last three lines are not package names, so install the
main packages directly:

```
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==2.2.4 scipy==1.15.2 scikit-learn==1.6.1 pandas==2.2.3 matplotlib==3.10.1 seaborn==0.13.2 pillow==11.1.0 tensorboard==2.19.0 torcheval==0.0.7 torchmetrics==1.7.0 progress==1.6
```

## Data

The images are not in this repository. They are FAA weather camera images from the AIR-VIEW
collection of the VizSim Lab at Ohio University (C. Mourning, Z. Wang and J. Murray, *AIR-VIEW: The
Aviation Image Repository for Visibility Estimation of Weather, a Dataset and Benchmark*,
arXiv:2506.20939, 2025).

The image folder has one subfolder per harvest, named by its time. Each file name gives the site,
the camera bearing and the reported visibility, for example
`2024-09-08-13-07-06/SITE19_ORNT325_VIS1-5mi.png` (1.5 mi).

- `splits/` lists the class-balanced subset used in the thesis, with paths relative to the image
  folder: 13,520 images from 344 sites (`train.csv` 9,460, `validation.csv` 2,030, `test.csv` 2,030).
- `splits/similarity/` holds the same lists without the two images whose camera views have no
  clear-day reference, and `reference_manifest.csv`, which records the reference chosen for each
  camera view.

The configuration files and the scripts' default paths point to `D:\Research - Lasya\NewWebcams`.
Before running anything, change `dset_path`, `splits_path`, `ref_dir` and `pretrained_path` in
`config.py`, and pass `--dset_path` to the scripts that take it.

## Run the experiments

**Pooled training.** `python main.py` takes no arguments and reads everything from `config.py`. The
`config.py` in this repository is the VisNet-Similarity run (a copy is in `VisNetSimilarity/`). For
VisNet and RMEP, change the network, the dataset module (`cls_10full`), the augmentation and the
normalization to the settings in Table 5.2 of the thesis. Checkpoints (`best_epochNN.pt`) and
TensorBoard logs are written under `runs/`, which is not committed.

**Per-site training.** The defaults match the thesis:
- only sites with at least 20 images are trained;
- each site is split 80/20 by capture event, and 20% of the training events are held out for
  validation;
- training runs for at most 60 epochs, with patience 15 and seed 42.

```
python trainpersite.py --use_all_splits --output_dir VisNetPerSiteAndPredictionsResults
python trainpersite.py --use_all_splits --pretrained_path runs/VisNetGlobal/best_epoch19.pt --output_dir VisNetFineTuneAndPredictionsResults
```

**Pooled models on the per-site test sets, and by site on the pooled splits:**

```
python eval_global_on_persite.py --checkpoint runs/VisNetGlobal/best_epoch19.pt --predictions VisNetPerSiteAndPredictionsResults/predictions.csv --output_dir VisNetGlobalOnPerSiteTest
python runsiteeval.py --checkpoint runs/VisNetGlobal/best_epoch19.pt --split test --output_dir VisNetSiteEvaluationResultsTest
python runsiteeval.py --checkpoint runs/VisNetGlobal/best_epoch19.pt --split validation --output_dir VisNetSiteEvaluationResultsValidation
```

**Clear-day references for VisNet-Similarity.** This writes one `<view>_reference` image per camera
view and `reference_manifest.csv` into a new folder:

```
python select_references.py --dset_path <images> --splits_path <images> --output_dir <images>/references_by_view --pool_path <images>
```

**Paired bootstrap.** `python bootstrap_ci.py` compares the six VisNet and RMEP configurations
(10,000 resamples, seed 42). `make_thesis_tables.py` repeats the comparison for all nine
configurations.

**LDA of the pooled embeddings.** `run_lda.ps1` lists every run in order. Each line is a
`make_lda_sites.py` or `make_lda_onesite.py` command that can also be run directly. Table 7.1 also
uses untrained runs with `--seed 1` and `--seed 2`.

**Dataset checks and example figures:**

```
python dataset_audit.py                                         # dataset_audit/
python dataset_audit.py --lowlight_scope all --out dataset_audit_all
python check_banners.py                                         # banner_check.csv
python make_example_figures.py --views SITE19_ORNT170           # figures_examples/
python make_example_figures.py --figures cameras
python datarange.py                                             # first and last harvest time (set ROOT first)
```

## Result folders

| Folder | Written by | Contents |
|---|---|---|
| `VisNetGlobalRunCSvs`, `RMEPGlobalrunCSVs`, `VisNetSimilarity` | `main.py` | Pooled runs: per-epoch metrics, splits, outputs |
| `*GlobalOnPerSiteTest` | `eval_global_on_persite.py` | Pooled models on the per-site test sets |
| `*PerSiteAndPredictionsResults` | `trainpersite.py` | Per-site models trained from scratch |
| `*FineTuneAndPredictionsResults` | `trainpersite.py --pretrained_path` | Per-site models fine-tuned from the pooled model |
| `*SiteEvaluationResultsValidation`, `*SiteEvaluationResultsTest` | `runsiteeval.py` | Pooled models on the pooled splits, by site |
| `dataset_audit`, `dataset_audit_all` | `dataset_audit.py` | Class counts and image brightness |
| `lda_results` | `make_lda_sites.py`, `make_lda_onesite.py` | LDA figures and summaries |
| `banner_check.csv` | `check_banners.py` | Position of the camera banner in the subset images |
| `thesis_tables`, `thesis_figures` | `make_thesis_tables.py`, `make_thesis_figures.py` | Tables and data figures of the thesis |
| `figures_examples` | `make_example_figures.py` | Example images in the thesis |

## Code provenance

The code builds on the VizSim Lab's code base at Ohio University. `code_provenance.csv`, written by
`compare_with_lab.py`, marks each file as new, modified, unchanged or only in the lab's version.
VisNet is from Palvanov and Cho (2019) and RMEP from Su et al. (2022). VisNet-Similarity
(`models/visnet/visnet_similarity.py`, `losses/similarity.py`, `dsets/webcams/cls_10full_dual.py`,
`select_references.py`) was written for this thesis.

## Citation

S. Mutukula, *Site-Specific Adaptation of Convolutional Neural Networks for Visibility Estimation
from Aviation Weather Camera Images*, M.S. thesis, Ohio University, 2026.
