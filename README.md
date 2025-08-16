# Reproduce figures from the preprint [Modeling hidden cognitive states reveals acute and chronic effects of fentanyl on decision-making](https://www.biorxiv.org/content/10.1101/2025.08.04.668448v1)

**Preprint:** bioRxiv, 2025. DOI: 10.1101/2025.08.04.668448

This repository contains code to (1) fit a Mixture-of-Agents Hidden Markov Model (MoA-HMM) to two-step task behavior and (2) reproduce the figures from the preprint.

## MoA-HMM Model Fitting
MoA-HMM jointly infers latent states and agent weights while supporting 5 agents (MF_c, MF_r, MB_c, MB, Bias) and 4 learning rates (one for each learning agent), with 1–7 hidden states.

## Quick start
### Clone and set paths
```
git clone https://github.com/Zhenlong-Zhang/MOA-HMM.git
cd MOA-HMM
```
### Prepare data (CSV per subject)
Each subject’s CSV must have the following columns:

| Column   | Meaning                                        |
|----------|------------------------------------------------|
| Choice   | 1 = left, 2 = right                           |
| Trans    | 0 = rare transition, 1 = common               |
| Reward   | 0 = no reward, 1 = reward                     |
| NewSess  | 1 = start of new session, 0 = continuation    |

**Example directory structure:**
```
data-cleaned/
├── group1/
│   ├── rat1.csv
│   ├── rat2.csv
│   └── ...
└── group2/
    ├── rat3.csv
    └── ...
```
## Citation
Zhang, Z., Janak, P. H., & Garr, E. (2025). *Modeling hidden cognitive states reveals acute and chronic effects of fentanyl on decision-making*. bioRxiv. https://doi.org/10.1101/2025.08.04.668448

## Contact

For questions or issues, please email:

- Zhenlong Zhang: zzl010824@gmail.com  
- Eric Garr: ericmgarr@gmail.com
