# MIMOSA
Please run setup.m to start.
## Pulseq Sequence
- Download Pulseq sequence programming environment (https://pulseq.github.io/) and add the matlab path.
- Run 01_gen_Pulse_Seq/3T/write_MIMOSA_1iso.m to generate Pulseq sequence file for 3T scan.
- Run 01_gen_Pulse_Seq/7T/write_MIMOSA_750um_iso_R4.m to generate Pulseq sequence file for 7T scan.
## Reconstrcution
The baseline zero-shot reconstrcution code is forked from https://github.com/byaman14/ZS-SSL & https://github.com/yohan-jun/Zero-DeepSub
### Installation
Dependencies are given in `02_Recon/3T/zsssl_recon_3T/environment_tf2.yml` and can be installed with``conda env create -f environment_tf2.yml``.
### Data
- The raw data of MIMOSA at R = 11.75 at 3T can be downloaded [here](https://www.dropbox.com/scl/fi/myo832a0xcuugjz8gcfyc/meas_MID00017_FID51769_MIMOSA_1iso_R11_d1_cplm_v2.dat?rlkey=gtorarrj9rup9c7n7k74l2kkz&st=gc388m7q&dl=0). After downloading the raw data, put it in the folder ``02_Recon/3T/rawdata``.
- The raw data of MIMOSA at R = 4 at 7T can be downloaded [here](https://www.dropbox.com/scl/fi/cxwcg2hrzxronrcruuz1y/meas_MID00608_FID210370_MIMOSA_TE60_4ms_T2prep8ms_750um_R4_fov240x232x192_uniform_ACS4d.dat?rlkey=vbukwdgexjgijre1iwkycxbr4&st=rrog9cks&dl=0). After downloading the raw data, put it in the folder ``02_Recon/7T/rawdata``
### Reconstrcution Pipeline
#### 1. Preprocessing
- Run `02_Recon/3T/prepare_data_for_zsssl_recon.m` and `02_Recon/7T/prepare_data_for_zsssl_recon.m` to prepare data for 3T and 7T scans, respectively.
#### 2. Training
- Run `02_Recon/3T/zsssl_recon_3T/zs_ssl_train_multi_mask_batch_v10_ms.py` and `02_Recon/7T/zsssl_recon_7T/zs_ssl_train_multi_mask_batch_v10_ms.py` to perform multi-contrast/-slice zero-shot self-supervised learning training for 3T and 7T scans, respectively. Prior to running training file, hyperparameters can be adjusted from parser_ops.py under the same path.
#### 3. Inference
- Run `02_Recon/3T/zsssl_recon_3T/zs_ssl_inference_ms.ipynb` and `02_Recon/7T/zsssl_recon_7T/zs_ssl_inference_ms.ipynb` to load the check points saved during training for 3T and 7T scans, respectively.
## Paramater Estimation
1. Run `03_ParamEstimation/3T/gen_MIMOSA_dict_3T.m` and `03_ParamEstimation/7T/gen_MIMOSA_dict_7T.m` to generate the dictionary for 3T and 7T scans, respectively.
2. Run `03_ParamEstimation/3T/MIMOSA_paramater_mapping_3T.m` and `03_ParamEstimation/7T/MIMOSA_paramater_mapping_7T.m` to perform paramater estimation process for 3T and 7T scans, respectively.
## Copyright & License Notice
This project is licensed for non-commercial, research use only. For other purposes, please contact ychen156@mgh.harvard.edu.

