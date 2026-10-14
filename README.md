# SMNDC1–MAPK3 TMA Computer Vision Analysis

This repository contains the image-processing, quantitative-analysis, statistical-analysis, survival-analysis, and figure-generation code used for paired Exon4 and NoExon4 pancreatic ductal adenocarcinoma tissue microarrays (TMAs).

The analysis focuses on two proteins measured in different cellular compartments:

- **MAPK3**, quantified within the cytoplasmic/tumor-cell compartment.
- **SMNDC1**, quantified within segmented nuclei.

Raw microscopy images, segmentation outputs, clinical annotations, and generated analysis results are not distributed with the source code.

## Repository structure

```text
.
├── README.md
├── SCRIPT_INDEX.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config/
│   └── paths.example.yaml
├── data/
│   └── README.md
├── src/
│   └── tma_if/
└── scripts/
    ├── 00_metadata/
    │   ├── build_core_manifest.py
    │   ├── diag_dataset_inventory.py
    │   └── qc_transforms_and_masks.py
    ├── 01_preprocessing/
    │   └── split_if_channels.py
    ├── 02_segmentation/
    │   └── segment_cellpose.py
    ├── 03_registration/
    │   ├── register_mapk3_smndc1.py
    │   └── plot_registration_qc.py
    ├── 04_quantification/
    │   ├── quantify_smndc1_nuclear.py
    │   └── quantify_ratios_and_trends.py
    ├── 05_analysis/
    │   ├── gaussian_parameters_ancova.py
    │   ├── nuclear_vs_cytoplasm.py
    │   └── protein_spot_distributions.py
    ├── 06_survival/
    │   ├── prepare_survival_inputs.py
    │   └── survival_delta_intervals.py
    └── 07_figures/
        ├── plot_mapk3_smndc1_relationships.py
        ├── plot_delta_boxplots.py
        ├── plot_density_cutoff_counts.py
        └── plot_representative_cores.py
```

## Image and channel conventions

Multichannel immunofluorescence images are separated into three analysis channels:

| Channel | Output label | Analysis role |
|---|---|---|
| C0 | B | nuclei |
| C1 | G | protein |
| C2 | R | cytoplasm / pan-cytokeratin |

Downstream scripts expect:

```text
CHAN_B_nucleosIF/
CHAN_G_proteinaIF/
CHAN_R_citoplasmaIF/
```

Channel separation is performed with:

```text
scripts/01_preprocessing/split_if_channels.py
```

The standard analysis outputs are 8-bit TIFF files:

```text
*_B_nucleos8.tif
*_G_proteina8.tif
*_R_citoplasma8.tif
```

Raw 16-bit channel images can optionally be retained in addition to the standard 8-bit outputs.

## Cellpose segmentation

Cell and nuclear segmentation is performed with **Cellpose**:

```text
scripts/02_segmentation/segment_cellpose.py
```

The retained workflow is parameterized for the SMNDC1 sections and operates on the B/G/R channel structure above. It produces nuclear labels, cytoplasmic/cell labels, and binary protein masks used by downstream analyses.

Typical mask outputs include:

```text
*_nuc_masks.tif
*_cp_masks.tif
*_prot_mask.tif
```

MAPK3 masks used by downstream scripts are treated as existing analysis inputs when generated separately.

## Core and patient identifiers

TMA core identifiers are normalized to a row-column representation such as:

```text
10_6
```

Equivalent historical forms may include:

```text
10-6
10_6.ome
```

Patient identifiers are stored as `CPID`.

Core-to-patient mapping is obtained from `TMA_Array.xlsx`. Clinical and survival annotations are obtained from `Tumor_Annotations.xlsx`.

These patient-level files are local inputs and should not be committed to a public source-code repository.

## Quantitative definitions

### MAPK3 cytoplasmic ratio

\[
\mathrm{ratio\_M\_cyto}
=
\frac{\mathrm{MAPK3\ positive\ area}}
{\mathrm{cytoplasmic\ area}}
\]

Typical variables include:

```text
area_cyto_px
mapk3_positive_area_px
ratio_M_cyto
```

### SMNDC1 nuclear ratio

\[
\mathrm{ratio\_S\_nuc}
=
\frac{\mathrm{SMNDC1\ positive\ nuclear\ area}}
{\mathrm{nuclear\ area}}
\]

The canonical area-based variable is:

```text
smndc1_positive_area_fraction
```

and is standardized downstream as:

```text
ratio_S_nuc
```

### SMNDC1 nuclear intensity

The nuclear quantification workflow also reports intensity-based measurements such as:

```text
smndc1_intensity_sum
smndc1_mean_intensity
```

These measurements are distinct from `ratio_S_nuc`.

Historical variables named `protein_rate_in_nucleus` should not be interpreted as equivalent to the positive-area fraction unless their definition has been verified for the specific source table.

## Exon4 versus NoExon4 comparisons

For paired patient-level analyses:

\[
\Delta = \mathrm{Exon4} - \mathrm{NoExon4}
\]

Therefore:

```text
Delta > 0   -> Exon4 is higher than NoExon4
Delta <= 0  -> NoExon4 is greater than or equal to Exon4
```

When a fold ratio is used:

\[
R = \frac{\mathrm{Exon4}}{\mathrm{NoExon4}}
\]

## Analysis populations

### Global

The **Global** population contains all eligible cells or nuclei retained by the corresponding quality-control and analysis filters.

### Registered

The **Registered** population contains observations retained after spatial registration and matching between MAPK3 and SMNDC1 sections.

### Standard SMNDC1

**Standard** SMNDC1 measurements are nuclear measurements obtained without restricting the analysis to MAPK3-matched registered nuclei.

`Global`, `Registered`, and `Standard` are not interchangeable terms.

## Analysis pipeline

### 1. Dataset inventory and metadata

```text
scripts/00_metadata/diag_dataset_inventory.py
scripts/00_metadata/build_core_manifest.py
scripts/00_metadata/qc_transforms_and_masks.py
```

These scripts inspect the dataset, build the canonical core/CPID manifest, and summarize mask and registration-transform availability.

### 2. IF channel separation

```text
scripts/01_preprocessing/split_if_channels.py
```

Separates multichannel IF images into nuclear, protein, and cytoplasmic channels using robust intensity scaling.

### 3. Cellpose segmentation

```text
scripts/02_segmentation/segment_cellpose.py
```

Runs Cellpose segmentation for nuclear and cytoplasmic/tumor-cell compartments and generates binary protein masks for the retained SMNDC1 workflow.

### 4. MAPK3–SMNDC1 registration

```text
scripts/03_registration/register_mapk3_smndc1.py
```

Registers MAPK3 sections to the corresponding SMNDC1 sections using cytoplasmic/pan-cytokeratin masks. The workflow includes multistart ECC registration, rotation search, coarse-to-fine optimization, optional phase-correlation initialization, and optional affine fallback.

```text
scripts/03_registration/plot_registration_qc.py
```

Recomputes overlap metrics in SMNDC1 space and generates registration quality-control tables and representative overlays.

### 5. Protein quantification

```text
scripts/04_quantification/quantify_smndc1_nuclear.py
```

Quantifies SMNDC1 nuclear morphology, intensity, and positive-area fraction for Standard and Registered/Coincident nuclear populations.

```text
scripts/04_quantification/quantify_ratios_and_trends.py
```

Constructs the canonical MAPK3 and SMNDC1 ratio tables, generates Global and Registered datasets, fits Exon4 and NoExon4 Gaussian/KDE distributions, and reports valid distribution intersections.

Canonical outputs include:

```text
mapk3_global.csv
smndc1_global.csv
registered_mapk3_smndc1.csv
distribution_parameters.csv
gaussian_intersections.csv
```

### 6. Statistical and morphological analyses

```text
scripts/05_analysis/gaussian_parameters_ancova.py
scripts/05_analysis/nuclear_vs_cytoplasm.py
scripts/05_analysis/protein_spot_distributions.py
```

These scripts compute patient-level distribution parameters, paired Exon4–NoExon4 comparisons, ANCOVA models, nuclear/cytoplasmic morphology relationships, and protein-positive spot distributions.

### 7. Survival analysis

```text
scripts/06_survival/prepare_survival_inputs.py
scripts/06_survival/survival_delta_intervals.py
```

These scripts generate harmonized survival inputs for MAPK3 and SMNDC1, Global and Registered populations, and perform Kaplan–Meier, log-rank, and Cox proportional-hazards analyses.

A Gaussian intersection is used only when a valid intersection exists or when an explicit cutoff is supplied.

### 8. Manuscript figures

```text
scripts/07_figures/plot_mapk3_smndc1_relationships.py
scripts/07_figures/plot_delta_boxplots.py
scripts/07_figures/plot_density_cutoff_counts.py
scripts/07_figures/plot_representative_cores.py
```

These scripts generate MAPK3–SMNDC1 relationship plots, Delta-stratified boxplots, Gaussian density/count figures, and representative core images.

## Statistical unit

The patient/CPID is used as the independent statistical unit for patient-level inference when multiple cells, nuclei, or cores contribute measurements to the same patient.

Cell- and nucleus-level measurements are used for visualization and descriptive distribution analyses, but individual cells or nuclei are not treated as independent patient-level observations unless clustering is explicitly modeled.

For paired Exon4/NoExon4 patient analyses, paired tests or equivalent tests of the within-patient difference are used.

Groups defined according to the sign of Delta are used for visualization or stratification; they are not compared inferentially using the same Delta variable that defined the groups.

## Configuration

Create a local configuration file from the example:

```bash
cp config/paths.example.yaml config/paths.yaml
```

Edit the local paths to match the analysis workstation.

Workstation-specific absolute paths should not be committed.

## Installation

The analysis was developed under Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Recorded software versions used during the analysis include:

```text
Python 3.10.18
Cellpose 3.1.1.2
NumPy 2.0.2
pandas 2.3.2
scikit-learn 1.7.2
seaborn 0.13.2
matplotlib 3.10.5
```

The remaining package versions should be recovered from the original analysis environment before producing a fully pinned reproducibility environment.

## Recommended execution order

```text
Dataset inventory and manifest
        |
        v
IF channel separation
        |
        v
Cellpose segmentation
        |
        v
MAPK3-SMNDC1 registration
        |
        v
registration quality control
        |
        v
SMNDC1 and MAPK3 quantification
        |
        v
Global and Registered analysis tables
        |
        v
statistical and morphological analyses
        |
        v
survival input preparation
        |
        v
Kaplan-Meier / Cox analysis
        |
        v
manuscript figures
```

Global analyses do not require MAPK3–SMNDC1 registration. Registered analyses require the registration and matching stages.

## Outputs

Generated outputs should be written outside the source-code tree or under ignored directories such as:

```text
outputs/
results/
logs/
```

Typical outputs include cell-level MAPK3 tables, nucleus-level SMNDC1 tables, registered MAPK3–SMNDC1 tables, core-level summaries, patient-level paired tables, Gaussian/KDE distribution parameters, registration QC tables, survival-analysis tables, and manuscript figures.

## Data availability and privacy

Raw microscopy data and patient-level clinical annotations are not included in this repository.

Clinical files should only be distributed according to the applicable study approvals and data-sharing policies.

## License

This code is distributed under the MIT License. See `LICENSE`.
