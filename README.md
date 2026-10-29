# MRI-BRIDGE: GENERATIVE DIFFUSION BRIDGE NETWORK FOR FAST MOTION-COMPENSATED ABDOMINAL MRI 

This repository contains code for data preprocessing, model training/inference, and evaluation of
the MRI-bridge. Due to ethical restrictions, we could one share
one test example that is uploaded, along with model weights, to zenodo under the 
doi: 10.5281/zenodo.21889000.

Please note, that some parts can't be performed exactly as in the paper, as they require the complete 
3D context or vendor-specific software.

## Repository Structure

```
MRI-bridge/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config.sh                           ← shell script where you define all paths
├── options.py                          ← option-file parser (adapted from UniDB-GOU)
├── train.py                            ← MRI-bridge training script
├── train_slab.py                       ← discriminative neural network (DiscNet) training script
├── training_utils.py                   ← helper functions for MRI-bridge and DiscNet trainings
├── inference.py                        ← inference script (MRI-bridge + DiscNet model)
├── calibration.py                      ← calibration script
├── options/                            ← YAML configuration files
│   ├── train.yml                       ← training config for MRI-bridge model
│   ├── train_slab.yml                  ← training config for DiscNet model
|   ├── inference.yml                   ← inference config for MRI-bridge and DiscNet for retrospectively accelerated test data, 
|   |                                     reference is XD-GRASP, R=50
|   ├── inference_R5.yml                ← inference config for MRI-bridge and DiscNet for retrospectively accelerated test data, 
|   |                                     reference is XD-GRASP, R=5
|   ├── inference_softgating.yml        ← inference config for MRI-bridge and DiscNet for retrospectively accelerated test data, 
|   |                                     reference is soft-gating, R=1
|   ├── inference_cal.yml               ← inference config for MRI-bridge for retrospectively accelerated calibration data, R=50
|   ├── inference_windows.yml           ← inference config for MRI-bridge for retrospectively accelerated test data with 9 
|   |                                     non-overlapping windows, R=50
|   ├── inference_single_full_slice.yml ← inference config for MRI-bridge, that was trained with single and full slices, for 
|   |                                     retrospectively accelerated validation data, R=5
|   ├── inference_slab_full_slice.yml   ← inference config for MRI-bridge, that was trained with slabs and full slices, for 
|   |                                     retrospectively accelerated validation data, R=5
|   └── inference_slab_patch.yml        ← inference config for MRI-bridge, that was trained with slabs and patches (final config), 
|                                         for retrospectively accelerated validation data, R=5
├── preprocessing/                      ← data splitting & preprocessing
|   ├── helpers.py                      ← helpers for split_data*.py scripts
|   ├── plotting_helpers.py             ← helpers to plot examples of split_data*.py scripts
│   ├── split_config_demo.py            ← exam-level split assignments (retrospectively accelerated example data)
│   └──split_data_demo.py               ← preprocessing & splitting for retrospectively accelerated example cases
└── evaluation/                         ← metrics, comparison & visualization tools
    ├── eval_helpers.py                 ← helpers for evaluation to plot results, etc.
    ├── metrics.py                      ← image quality metrics for retrospective cases, where reference is given 
    ├── plot_rater_results.py           ← reader-study score visualization (boxplots)
    ├── plot_results.py                 ← visualize PURE-corrected outputs per slice
    └── SNR_analysis.py                 ← SNR analysis using sliding-window results
```

## Model Code

The neural network architecture is based on [UniDB (Xue et al.)](https://github.com/UniDB-SOC/UniDB).
Our modifications (custom training scripts, slab model, MRI-specific data loading, inference pipeline) are maintained in a 
separate fork:

> **Fork:** <https://github.mskcc.org/schellem/UniDB-fork>

The training and inference scripts expect the fork to be cloned as a sibling directory:

```
parent_dir/
├── MRI-bridge/          ← this repository
└── UniDB-fork/
    └── UniDB-GOU/       ← forked model code (models, utils, data loaders)
```

```bash
# Clone both repositories side-by-side
git clone https://github.mskcc.org/schellem/MRI-bridge.git
git clone https://github.mskcc.org/schellem/UniDB-fork.git
```

> **Note:** The training and inference scripts (`train.py`, `train_slab.py`, `inference.py`) import model architectures, 
> utilities, and data loaders from `../UniDB-fork/UniDB-GOU/`.  The local `options.py` adapts the original
> option parser so that experiment artifacts (checkpoints, logs, results) are stored under `MRI-bridge/` 
> rather than inside the fork's directory tree.

## Quick Start

### 1. Set the following variables in the /parent_dir/MRI-bridge/config.sh:

```bash
export PYTHONPATH=${PYTHONPATH}:/parent_dir         # Path of parent_dir
export DEMO_PATH=/path/to/data                      # Path to folder with data, models, etc. 
export DATAFOLDER_NAME=name_of_input_data_folder    # name of folder that includes image data (NUFFT, XD-GRASP, etc.), e.g. Demo_data
export RUN_NAME=name_of_run                         # basic name that will be first part of each created results-foldername
export EXPERIMENT_NAME=experiment_name              # name of results folder
export CALIBRATION_LAMBDA=dummy_calibration_lambda  # dummy value, e.g., "2.0", shall be replaced after calibration
export PROVIDED_RESULTS='Provided_Results'          # name of folder including DICOMs and segmentations, that were provided for this demo
```

```bash
source "/parent_dir/MRI-bridge/config.sh"
```
### 2. Install dependencies

```bash
conda create --name MRIbridge python=3.9
conda activate MRIbridge
# install requirements
pip install -r requirements.txt
# install PyTorch with the suitable cuda version from https://pytorch.org/get-started/previous-versions/. For example:
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# Note that dependent on your hardware, newer versions for torch and the packages in requirements.txt might be suitable. 
# Code was tested with the above versions only.
 
# For metrics evaluation:
git clone https://github.com/Bayer-Group/mr-image-metrics.git
pip install -e /path/to/mr-image-metrics
```

### 3. Preprocess & split data

Plase note, that this demo only provides very limited data. More specifically, five neighboring test slices are shared to run inference for 
one slice and three post-processed slices (one validation slice, one calibration slice, and one test slice) are shared for analysis.
However, this serves only for demonstration and it's obvious that you need more data for training, validation, calibration, testing, and analysis. 
The shape and structure of the data can be extracted from the shared exam in ```$DEMO_PATH/$DATAFOLDER_NAME/```.

The following script loads the images (NUFFT, XD-GRASP, etc.) of ```$DEMO_PATH/$DATAFOLDER_NAME``` according to the ```preprocessing/split_config_*.py```.
It performs preprocessing steps such as normalization, cropping, and rotations and saves the files (if --save) 
as well as the datasplit.pkl under ```$DEMO_PATH/$RUN_NAME``` and/or ```$DEMO_PATH/${RUN_NAME}_windows```. For more details, see ```split_data_*.py```. 
If --plot is set, some examples are plotted under ```$DEMO_PATH/${RUN_NAME}_plots```. 

```bash
cd /parent_dir/MRI-bridge/preprocessing/

# Standard mode (retrospective cohort), extracting NUFFT of first 20s, XD-GRASP reference, and soft-gating reference
python split_data_demo.py --mode standard_20s --save --plot
# Window mode (retrospective cohort), extracting NUFFTs of any 20s window, XD-GRASP reference, and soft-gating reference
python split_data_demo.py --mode window --save
```

### 4. Train the model
Assuming you have training data, adjust 
1. options/train.yml for the MRI-bridge and
2. options/train_slab.yml for the discriminative U-Net (comparison model)

Then run: 
```bash
cd /parent_dir/MRI-bridge/

# MRI-bridge
python train.py -opt options/train.yml
# DiscNet
python train_slab.py -opt options/train_slab.yml
```

### 5. Inference on retrospectively accelerated cases

- Note, that the pre-trained models saved under ```$DEMO_PATH/models``` are used here. 
If you want to use your self-trained models, replace them there or adjust the configuration yaml files accordingly.

- For each evaluation, there is a ```config*.yml``` file used. If you want to change something, please adapt the config file. 

- For each evaluation, the MRI-bridge and the DiscNet are applied, although the latter is sometimes not needed.

- For each evaluation, you should see some plots in this repository under ```/results/${RUN_NAME}_suffix``` (suffix depends on the inference config, e.g., "_cal", "_50_windows", etc.). In addition, the results are saved as \*.mat files in the ```$DEMO_PATH/$EXPERIMENT_NAME/${RUN_NAME}_suffix```.

#### 5.1 Inference on calibration set

Assuming you have calibration data, run inference of the MRI-bridge (50 inferences) and the DiscNet on the held-out calibration set via:

```bash
cd /parent_dir/MRI-bridge/
python inference.py -opt options/inference_cal.yml
```

#### 5.2 Inference on test set

Run inference of the generative MRI-bridge network (50 and 5 inferences per slice) and the DiscNet on the held-out test set via:
```bash
cd /parent_dir/MRI-bridge/
python inference.py -opt options/inference.yml
python inference.py -opt options/inference_R5.yml
```

To provide the softgating reconstruction as the comparison method instead of the discriminative neural network, please run:
```bash
cd /parent_dir/MRI-bridge/
python inference.py -opt options/inference_softgating.yml
```
Note, that here the MRI-bridge was run with 1 inference only. 


In addition, inference is performed with 50 inferences for 9 non-overlapping windows via:

```bash
cd /parent_dir/MRI-bridge/
python inference.py -opt options/inference_windows.yml
```

#### 5.3 Inference on validation set

As part of the hyperparameter tuning, the best training mode of the MRI-bridge was determined based on the validation set. To showcase the differences, inference for different training modes (single and full slice training vs. slab and full slice training vs. slab and patch-wise training) is performed on the validation set, assuming you have some, via:

```bash
cd /parent_dir/MRI-bridge/
python inference.py -opt options/inference_single_full_slice.yml

python inference.py -opt options/inference_slab_full_slice.yml

python inference.py -opt options/inference_slab_patch.yml
```

#### 5.4 Postprocessing of retrospectively accelerated results

Semantic segmentation of the body trunk and of a variety or organs are used for reconstruction calibration and analysis. For this, the GE scanner reconstructions corresponding to the MRI-bridge slices were extracted. In addition, our results were corrected for coil profile variations and DICOMs were created. 

This part of the code is up to the user. 

However, our DICOMs for the example data can be found under: 
```bash
$DEMO_PATH/Provided_Results/PURE_Results
$DEMO_PATH/Provided_Results/PURE_Results_5inferences
$DEMO_PATH/Provided_Results/PURE_Results_50_windows
$DEMO_PATH/Provided_Results/PURE_Results_single_full_slice
$DEMO_PATH/Provided_Results/PURE_Results_slab_full_slice
$DEMO_PATH/Provided_Results/PURE_Results_slab_patch
```
Note, that all following steps will use those results. If you want to work with your own results, please adapt the ```PROVIDED_RESULTS``` in the config file.

#### 5.5 Semantic segmentation for analysis

Semantic segmentations of the body trunk were performed manually in 3DSlicer on the GE scanner DICOMs.
For semantic segmentation of a variety of organs, TotalSegmentator was applied to the GE scanner DICOMs in "total_MR" mode. If you want to run TotalSegmentator, please see https://github.com/wasserth/totalsegmentator.

The semantic segmentations for the example data can be found under:
```$DEMO_PATH/Provided_Results/Segmentations```
Note, that some of the following steps will use those segmentations. If you want to work with your own results, please adapt the ```PROVIDED_RESULTS``` in the config file.

#### 5.6 Calibration

The calibration can be performed on the one PURE corrected calibration slice within the body trunk via:

```bash
cd /parent_dir/MRI-bridge/
python calibration.py
```

Copy the final calibration lambda value to ```/parent_dir/MRI-bridge/config.sh```. Note, that this only makes sense with more calibration files.

### 6. Plot results

Example slices can be plotted via: 

```bash
cd /parent_dir/MRI-bridge/evaluation
python plot_results.py
```
Results are saved in ```$DEMO_PATH/$EXPERIMENT_NAME``` under 
```
${RUN_NAME}_Plots_PURE_Results
${RUN_NAME}_Plots_PURE_Results_single_full_slice
${RUN_NAME}_Plots_PURE_Results_slab_full_slice
${RUN_NAME}_Plots_PURE_Results_slab_patch
```

### 7. SNR analysis

An SNR analysis is performed on the sliding-window results via: 

```bash
cd /parent_dir/MRI-bridge/evaluation
python SNR_analysis.py
```

Results are saved in ```$DEMO_PATH/$EXPERIMENT_NAME``` under ```${RUN_NAME}_Plots_PURE_Results_50_windows```.

### 8. Compute metrics

Metrics are computed and plotted via:

```bash
cd /parent_dir/MRI-bridge/evaluation
python metrics.py --compute_metrics
```
Results are saved in ```$DEMO_PATH/$EXPERIMENT_NAME``` under ```${RUN_NAME}_Metrics```.

### 9. Reader study

If you have reader scores, you can plot those via;
```bash
cd /parent_dir/MRI-bridge/evaluation
python plot_rater_results.py
```
Results are saved in ```$DEMO_PATH/$EXPERIMENT_NAME```.

## Citation

If you use this code, please cite:

- **Model architecture:** [UniDB (Xue et al.)](https://github.com/UniDB-SOC/UniDB)
- **This work:** [Schellenberg M, Do R, Otazo R. Generative diffusion bridge reconstruction for accelerated motion-compensated free-breathing abdominal MRI. In: Proceedings of the ISMRM annual meeting, 2026.]

## License

Model code (UniDB-GOU/) is subject to the [UniDB license](https://github.com/UniDB-SOC/UniDB/blob/main/LICENSE) and (MRI-bridge/) is subject to the [MRI-bridge license](https://github.mskcc.org/MRI-Physics/MRI-bridge/blob/main/LICENSE).
