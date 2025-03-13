# MIMOSA
## 1. Pulseq Sequence
Please reach out to ychen156@mgh.harvard.edu for the script to generate .seq file for MIMOSA.
## 2. Reconstrcution
The baseline zero-shot reconstrcution code is forked from https://github.com/byaman14/ZS-SSL & https://github.com/yohan-jun/Zero-DeepSub
### Installation
Dependencies are given in Recon/3T/zsssl_recon_3T/environment_tf2.yml, which can be installed with``conda env create -f environment_tf2.yml``.
### Data
1. The raw data of MIMOSA at R = 11.75 at 3T can be found [here](https://www.dropbox.com/scl/fi/myo832a0xcuugjz8gcfyc/meas_MID00017_FID51769_MIMOSA_1iso_R11_d1_cplm_v2.dat?rlkey=gtorarrj9rup9c7n7k74l2kkz&st=gc388m7q&dl=0).
2. The raw data of MIMOSA at R = 4 at 7T can be found here.
### Reconstrcution Pipeline
1. Run prepare_data_for_zsssl_recon.m
2. Run zs_ssl_train_multi_mask_batch_v10_ms.py to perform multi-contrast/-slice zero-shot self-supervised learning training. Prior to running training file, hyperparameters can be adjusted from parser_ops.py.
3. Run zs_ssl_inference_ms.ipynb to load the check points saved during training.
## 3. Paramater Estimation
1. Run gen_MIMOSA_dict_3T/7T.m to generate the dictionary.
2. Run MIMOSA_paramater_mapping_3T/7T.m to perform paramater estimation process.

