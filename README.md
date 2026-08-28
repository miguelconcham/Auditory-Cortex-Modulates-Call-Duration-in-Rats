# Auditory cortex modulates call duration in rats

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42003--026--09608--9-blue)](https://doi.org/10.1038/s42003-026-09608-9)
[![Paper](https://img.shields.io/badge/Communications%20Biology-2026-green)](https://www.nature.com/articles/s42003-026-09608-9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MATLAB analysis code accompanying:

> Tang, W., Concha-Miranda, M. & Brecht, M. Auditory cortex modulates call duration in rats. *Communications Biology* **9**, 353 (2026). [https://doi.org/10.1038/s42003-026-09608-9](https://doi.org/10.1038/s42003-026-09608-9)

Wei Tang and Miguel Concha-Miranda contributed equally. Corresponding author: [Michael Brecht](mailto:michael.brecht@bccn-berlin.de).

## Overview

This repository contains the MATLAB scripts used to generate the electrophysiology and white-noise behavior figures in the paper. The study shows that rat auditory cortex is not only sensory: distinct functional cell types respond differently during self-generated calls versus playback, onset-suppressed neurons predict upcoming call duration and occurrence, and bidirectional pharmacological or noise-driven recruitment of auditory cortex changes total call duration.

The scripts cover:

- Classification of auditory cortex neurons recorded with Neuropixels during PAG-evoked vocalizations
- Call versus playback PSTHs, laminar depth, rate–duration correlations, and SVM prediction of call occurrence
- Effects of in-ear white noise on call duration, frequency, loudness, and call number
- Timing of noise relative to call onset (pre-call vs during-call; Supplementary Fig. 7)

Pharmacological results (Fig. 3 and Supplementary Fig. 5; muscimol / gabazine) were analyzed in GraphPad Prism 8. Source data for those panels are provided as **Supplementary Data 1** with the paper, not as MATLAB scripts here.

## Repository layout

```text
.
├── Behavior Analysis/
│   ├── Behavior_Analysis_main_figure.m          # Fig. 4 and Supplementary Fig. 6
│   └── latenci2noise_SupplementaryFigure.m      # Supplementary Fig. 7
├── Ephys Analysis/
│   ├── Neuron_classification.m                  # cell-type classification (Supp. Fig. 1)
│   ├── Ploting_Ephys_data.m                     # Figs. 1–2 and related supplements
│   └── Mixed_model_correlation_lenght_firing_rate.m  # mixed-effects rate vs duration
├── LICENSE
└── README.md
```

Filenames match the files in the repository (including original spelling).

## Requirements

Analyses in the paper were run in **MATLAB R2023b** (The MathWorks). Later releases should work; older versions may fail on `fitlme` / `omitmissing` usage.

Required MathWorks products:

| Product | Used for |
| --- | --- |
| MATLAB | PSTHs, figures, table I/O |
| Statistics and Machine Learning Toolbox | `fitlm`, `fitlme`, `anovan`, `ttest`, `signrank`, `kstest`, Spearman/Pearson `corr` |

Place helper functions that the scripts call on the MATLAB path before running:

| Function | Called from |
| --- | --- |
| `assignBrainArea` | `Neuron_classification.m`, `Ploting_Ephys_data.m` |
| `assign_depth` | same |
| `CallCount` | `Behavior_Analysis_main_figure.m` |

## Data

The paper’s data-availability statement points to this repository for source data other than Fig. 3 / Supplementary Fig. 5. Scripts assume those files sit **in the same folder as the script that loads them** (or at a path you set in the first section). GitHub does not currently host the large `.mat` Neuropixels datasets; copy them into the matching analysis folder after download.

### Electrophysiology (`Ephys Analysis/`)

| File / pattern | Loaded by |
| --- | --- |
| `*DataSet.mat` | `Neuron_classification.m`, `Ploting_Ephys_data.m` |
| `NeuronTypesAfterRevision.xlsx` | same |
| `summary length correlation.xlsx` | `Ploting_Ephys_data.m` |
| `CorrectedBoxes_Stats.xlsx` | `Ploting_Ephys_data.m` (dataset 4 call boxes) |
| `table_r2_withds4.mat` (variable `table_r2_withds4`) | `Mixed_model_correlation_lenght_firing_rate.m` |

Each `*DataSet.mat` file is expected to contain a `DataSet` struct with at least:

- `spike_times`, `spike_clusters`, `good_clusters`, `mua_clusters`
- `ResponseTypes`, `CallStats`, `STIM_TIMES`
- `NeuropixelsDepth`, `synch_models.AUDIO_NPX`
- optional `NPX_type` (1 or 2) and `y_pos` for Neuropixels 2.0 depth assignment

Spike times are converted to seconds with a 30 kHz sampling rate and aligned to audio time via `predict(DataSet.synch_models.AUDIO_NPX, ...)`.

### Behavior (`Behavior Analysis/`)

| File | Loaded by |
| --- | --- |
| `Behavior DATA.mat` (variable `ALL_CALLS_TOGETHER`) | `Behavior_Analysis_main_figure.m` |
| `synch_model_spike2audio.mat` | `latenci2noise_SupplementaryFigure.m` |
| `Noise offset and onset.xlsx` | same |
| `merged_audio 2025-05-06 12_13 PM_Stats.xlsx` | same |
| Spike2 stim file named `mc250207_3 exp3` | same (see path note below) |

Fig. 3 / Supplementary Fig. 5 source tables: Supplementary Data 1 on the [publisher page](https://www.nature.com/articles/s42003-026-09608-9).

## How to run

1. Install MATLAB (R2023b or later) with the Statistics and Machine Learning Toolbox.
2. Put the data files listed above next to the script that loads them.
3. Add any helper functions (`assignBrainArea`, `assign_depth`, `CallCount`) to the MATLAB path.
4. `cd` into the analysis folder, then run the script.

```matlab
% Electrophysiology
cd('Ephys Analysis')
Neuron_classification          % iterative call vs playback classification
Ploting_Ephys_data             % main ephys figures (long; many sections)
Mixed_model_correlation_lenght_firing_rate

% White-noise behavior
cd('../Behavior Analysis')
Behavior_Analysis_main_figure
```

Scripts are organized in `%%` sections. You can run them top-to-bottom or section by section. Several blocks open figures and call `pause`; close windows or press a key in the Command Window to continue.

### Path that must be edited

`latenci2noise_SupplementaryFigure.m` currently sets a lab file-server path at the top of the file:

```matlab
behavior_data = '\\experimentfs.bccn-berlin.pri\...\Behavior Analysis';
```

Point `behavior_data` at the local folder that contains the Spike2 stim file (`mc250207_3 exp3`) before running. Excel tables and `synch_model_spike2audio.mat` are still loaded from the current working directory.

## Script-to-figure map

| Paper figure | Script | What it does |
| --- | --- | --- |
| Fig. 1, Supplementary Figs. 1–2 | `Ephys Analysis/Ploting_Ephys_data.m` | Onset/offset PSTHs for call and playback, call−playback difference, example neurons, depth distributions |
| Supplementary Fig. 1 (classification) | `Ephys Analysis/Neuron_classification.m` | Iterative z-score classification of cortical units into pre-call, onset, ramping, and non-responsive groups |
| Fig. 2, Supplementary Figs. 3–4 | `Ephys Analysis/Ploting_Ephys_data.m` | Population rate vs call duration, SVM features for call occurrence, example rasters |
| Fig. 2D / mixed-model R² | `Ephys Analysis/Mixed_model_correlation_lenght_firing_rate.m` | Linear mixed-effects models (`rate ~ CallLength + (1\|Ds)`) per cell type |
| Fig. 3, Supplementary Fig. 5 | — | GraphPad Prism; see Supplementary Data 1 |
| Fig. 4, Supplementary Fig. 6 | `Behavior Analysis/Behavior_Analysis_main_figure.m` | Paired noise vs baseline call trains; Spearman correlations of duration, frequency, and amplitude with noise level (0 to −40 dB) |
| Supplementary Fig. 7 | `Behavior Analysis/latenci2noise_SupplementaryFigure.m` | First-call duration and frequency vs noise onset/offset latency |

Cell types used in the ephys scripts correspond to the paper’s five functional classes (codes `A`, `B`, `D`, and non-responsive `N`/`T` in `NeuronTypesAfterRevision.xlsx`):

- Pre-call activated
- Onset activated / onset suppressed
- Ramping activated / ramping suppressed
- Non-responsive

Onset-suppressed neurons are the population whose pre-call firing predicts upcoming call duration (Fig. 2).

## Methods snapshot

Details are in the paper. In brief:

- Male Long-Evans rats; urethane anesthesia; PAG electrical stimulation to evoke 20–35 kHz call sequences.
- Neuropixels 1.0/2.0 in auditory cortex; spikes sorted with Kilosort 2.0 and curated in Phy.
- Calls detected with [DeepSqueak](https://github.com/DrCoffey/DeepSqueak) v3.
- Playback of the animal’s own calls compared with self-generated calls; white noise delivered in-ear at 0, −10, −20, −30, and −40 dB relative to 75 dB.

## Citation

If you use this code or the associated data, please cite:

```bibtex
@article{Tang2026auditory,
  title   = {Auditory cortex modulates call duration in rats},
  author  = {Tang, Wei and Concha-Miranda, Miguel and Brecht, Michael},
  journal = {Communications Biology},
  volume  = {9},
  pages   = {353},
  year    = {2026},
  doi     = {10.1038/s42003-026-09608-9}
}
```

Related brainstem mapping from the same group: Concha-Miranda, M., Tang, W., Hartmann, K. & Brecht, M. *J. Neurosci.* 42, 8252–8261 (2022). [https://doi.org/10.1523/JNEUROSCI.0813-22.2022](https://doi.org/10.1523/JNEUROSCI.0813-22.2022)

## License

Analysis code in this repository is released under the [MIT License](LICENSE). The article is open access under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

Questions about the experiments or code: [Michael Brecht](mailto:michael.brecht@bccn-berlin.de) (corresponding author) or open a GitHub issue on this repository.
