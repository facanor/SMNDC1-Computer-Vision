# Data directory

This directory is reserved for local analysis inputs and should not contain patient-level or raw microscopy data in the public repository.

The cleaned repository is designed so that code is version-controlled while study data remain outside Git or are copied locally into ignored paths when needed.

## Expected local inputs

Typical inputs include:

```text
TMA_Array.xlsx
Tumor_Annotations.xlsx
```

and the corresponding microscopy-derived image and mask directories.

These files are not distributed with the repository.

## TMA metadata

`TMA_Array.xlsx` is used to map TMA core positions or normalized core identifiers to patient identifiers (`CPID`).

Core identifiers are normalized to a row-column form such as:

```text
10_6
```

Historical alternatives such as `10-6` or names containing `.ome` may be normalized by the metadata utilities.

## Clinical and survival annotations

`Tumor_Annotations.xlsx` contains patient-level clinical and survival information used by downstream analyses.

Relevant columns may include:

```text
CPID
Date_of_diagnosis
Date_merged_last_followup_death
OS_censor
```

The exact column names expected by the local analysis environment can be configured in:

```text
config/paths.yaml
```

## Image data

The analysis expects multichannel immunofluorescence images organized into MAPK3 and SMNDC1, Exon4 and NoExon4 datasets.

After preprocessing, the standard channel directories are:

```text
CHAN_B_nucleosIF/
CHAN_G_proteinaIF/
CHAN_R_citoplasmaIF/
```

with:

```text
B -> nuclei
G -> protein
R -> cytoplasm / pan-cytokeratin
```

## Segmentation and mask inputs

Downstream analyses may use masks such as:

```text
*_nuc_masks.tif
*_cp_masks.tif
*_prot_mask.tif
```

where:

```text
nuc_masks  -> labeled nuclear masks
cp_masks   -> labeled cell/cytoplasmic masks
prot_mask  -> binary protein-positive mask
```

## Canonical derived tables

The cleaned workflow generates the following canonical analysis tables:

```text
nuclei_features_standard.csv
nuclei_features_coincident.csv

mapk3_global.csv
smndc1_global.csv
registered_mapk3_smndc1.csv

distribution_parameters.csv
gaussian_intersections.csv
```

These are generated outputs and should normally be written under an ignored `results/` or `outputs/` directory rather than committed to the repository.

## Data privacy

Patient-linked files, clinical annotations, raw microscopy data, and identifiable or potentially re-identifiable study data should only be stored and shared according to the applicable institutional approvals and data-sharing policies.

The public code repository should contain only source code, configuration examples, and documentation.
