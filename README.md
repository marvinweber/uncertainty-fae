# Uncertainty Quantification in Forensic Age Estimation

(!) WIP

## Directory Structure
- `config`: (Example) config files.
- `notebooks`: Notebooks with examples and or smaller evaluations, etc.
- `scripts`: Python scripts for training, evaluation, etc.
- `slurm_commands`: Scripts and Description for Slurm Job Usage (for training and
  evaluation) for the FAE framework.
- `src`: Main/ Source-Code for the Uncertainty Framework.
  - `src/clavicle_ct`: Code for the *Clavicle CT* dataset.
  - `src/rsna_boneage`: Code for the *RSNA Bone Age* dataset.
  - `src/swa_gaussian`: Base code for SWAG implementation and PyTorch Lightning
    integration (see `swa_gaussian/README.md` for details).
  - `src/uncertainty_fae`: Code for the generic uncertainty part/ framework
    (e.g., interfaces, evaulation code, and so on...).
  - `src/uncertainty_fae/util`: Utility methods and classes (mainly used by or
    helful for usage with the uncertainty framework).
- `util`: Utility scripts; see `util/README.md` for details.

## Docker Setup
TBD

## Notebook Usage
If you want to write and/or use the notebooks, you should install the
`ipykernel` and `ipywidgets` packages.

## Uncertainty Framework
The `uncertainty_fae` together with the provided `scripts` can be seen as a
lightweight "framework" to conduct uncertainty experiments in the context of
age estimation (probably also for similar, regression based tasks).

The package defines a few interfaces and utility classes/methods, that help to
implement and test UQ integrations.

## Datasets

### RSNA Bone Age
Source:
[RSNA Pediatric Bone Age Challenge](https://pubs.rsna.org/doi/10.1148/radiol.2018180736)  
Download: [Dataset and Annotations](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017)

TBD

### Clavicle CT
Unfortunately, the **Clavicle CT** dataset is an internal data set of the
University Hospital of LMU Munich, hence, it cannot be provided publicly.

The architecture (network, LitModels and UQ integration) can still be inspected
from the `src/clavicle_ct` package.

## Scripts

### Training
TBD

## Evaluation
TBD
