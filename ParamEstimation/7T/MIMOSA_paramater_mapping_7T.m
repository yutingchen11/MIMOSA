addpath(genpath('utils/'))
%% load dict
load('dict/dict_mimosa_TE60p4_B1_140_v1.mat');
load('dict/ielookup_4qalas_IR750.mat');
%--------------------------------------------------------------------------
%% set qalas param 
%--------------------------------------------------------------------------

param.esp             = 6.3 * 1e-3;
param.turbo_factor    = 115;%127 for 1x1x3mm;128 for 1x1x4mm

param.TR          = 3600e-3 - 6.3*115e-3 + 115*27.5*1e-3 - 162.4*2e-3;
param.alpha_deg   = 4;
param.num_reps    = 5;
param.echo2use    = 1;
param.gap_between_readouts    = 900e-3;
param.time2relax_at_the_end   = 0;

param.nconsrast =7;%nacq

% ##### mgre
param.TE_mte = [3:7:25].*1e-3;
param.TR_mte = 27.5e-3;
param.esp_mte = 7e-3;
nechoes = length(param.TE_mte);


TR                      = param.TR;
num_reps                = 5;
echo2use                = 1;
gap_between_readouts    = 900e-3;
time2relax_at_the_end   = 0;
alpha_deg = 4;
esp             = param.esp;
turbo_factor    = param.turbo_factor;

TR_mte = param.TR_mte;
esp_mte = param.esp_mte;
TEs = param.TE_mte;
nechoes = length(TEs);

img = abs(img_zsssl);
T1_all = zeros(size(img_zsssl,1),size(img_zsssl,2),size(img_zsssl,3));
T2_all = zeros(size(img_zsssl,1),size(img_zsssl,2),size(img_zsssl,3));
PD_all = zeros(size(img_zsssl,1),size(img_zsssl,2),size(img_zsssl,3));
T2s_all = zeros(size(img_zsssl,1),size(img_zsssl,2),size(img_zsssl,3));
IE_all = zeros(size(img_zsssl,1),size(img_zsssl,2),size(img_zsssl,3));


%--------------------------------------------------------------------------
%% Load b1 map and resize to qalas, for spetial reduced B1 slices, 42, for zju prisma B1map product
%--------------------------------------------------------------------------
matrix_size = [size(img,1),size(img,2),size(img,3)];

img_b1_load = niftiread('data\B1_map_Head_tra_fa8_20241126194221_4_Eq_1.nii');
img_b1 = double(img_b1_load)/800;       % reference fa 800

img_b1 = permute(img_b1, [2 1 3]); 

img_b1 = flip(img_b1,2); % check orientation to match the raw data
img_b1 = imresize3(img_b1, matrix_size); %  to match img FOV and size

imagesc3d2(img_b1, s(img_b1)/2 + [0 40 0], 50, [0 0 0], [0,1.5]);colormap jet


N = [size(img, 1) size(img, 2) size(img, 3)];

%% gen mask
% using FSL BET to generate brain mask, 
voxel_size = [0.75 0.75 0.75];
mask_brain = BET(rsos(img,4),matrix_size,voxel_size,0.5,0);
imagesc3d2( mag.*mask_brain, s(mag.*mask_brain)/2, 5, [0,0,0], [0 0.01])

msk = mask_brain;
%--------------------------------------------------------------------------
%% threshold high and low B1 values: use raw b1 map without polyfit
%--------------------------------------------------------------------------

thre_high   = 1.4;
thre_low    = 0.01;

temp        = img_b1 .* msk;

temp(temp > thre_high)  = thre_high;
temp(temp < thre_low)   = thre_low;

temp        = temp .* msk;
img_b1      = temp .* msk;


%--------------------------------------------------------------------------
%% create masks for each b1 value
%--------------------------------------------------------------------------

num_b1_bins = 140; % default: 50

b1_val      = linspace( min(img_b1(msk==1)), max(img_b1(msk==1)), num_b1_bins );
%b1_val      = 1; % no b1 correction
sum_msk     = sum(msk(:));

if length(b1_val) == 1
    % do not use b1 correction
    msk_b1 = msk;
else
    
    msk_b1 = zeross([N,length(b1_val)]);
    
    for t = 1:length(b1_val)
        if t > 1
            msk_b1(:,:,:,t) = (img_b1 <= b1_val(t)) .* (img_b1 > b1_val(t-1));
        else
            msk_b1(:,:,:,t) = msk.*(img_b1 <= b1_val(t));
        end
        
        percent_bin = sum(sum(sum(msk_b1(:,:,:,t),1),2),3) / sum_msk;
        
        if t == length(b1_val)
            msk_b1(:,:,:,t) = img_b1 > b1_val(t-1);
        end
        
        msk_b1(:,:,:,t) = msk_b1(:,:,:,t) .* msk;
    end
end

%% estimate T2* separatly using vapro
image_series = abs(squeeze(img(:,:,:,4:end)));
TE = TEs*1e3;
[Np, Nf, Nfr,~] = size(image_series);
Nsize = [Np, Nf, Nfr];
possibleT2Values = [1:500]; % the range of possible T2 values
[T2std, Mstd] = t2EstiVAPRO(reshape(image_series, Np*Nf*Nfr, [])', TE, possibleT2Values); 
Mstd  = reshape(Mstd, Np, Nf,Nfr);
T2std = reshape(T2std, Np, Nf,Nfr);

%% show the distribution of T2* 

msk = T2std>1;
% msk = logical(mask2);
A_flat = T2std(msk);
data = A_flat(:);
values = unique(data); %
counts = histc(data(:), values); % 

figure;
plot(values, counts);

%% 
%--------------------------------------------------------------------------
% threshold high and low T2* values: use T2* map
%--------------------------------------------------------------------------

thre_high   = max(values);
thre_low    = min(values);

temp        = T2std .* msk;

temp(temp > thre_high)  = thre_high;
temp(temp < thre_low)   = thre_low;

temp        = temp .* msk;
T2std      = temp .* msk;


%--------------------------------------------------------------------------
% create masks for each T2s value
%--------------------------------------------------------------------------


T2s_val      = values;

sum_msk     = sum(msk(:));

if length(T2s_val) == 1

    msk_t2s = msk;
else
    
    msk_t2s = zeross([N,length(T2s_val)]);
    
    for t = 1:length(T2s_val)
        if t > 1
            msk_t2s(:,:,:,t) = (T2std <= T2s_val(t)) .* (T2std > T2s_val(t-1));
        else
            msk_t2s(:,:,:,t) = msk.*(T2std <= T2s_val(t));
        end
        
        percent_bin = sum(sum(sum(msk_t2s(:,:,:,t),1),2),3) / sum_msk;
        
        if t == length(T2s_val)
            msk_t2s(:,:,:,t) = T2std > T2s_val(t-1);
        end
        
        msk_t2s(:,:,:,t) = msk_t2s(:,:,:,t) .* msk;
    end
end

%--------------------------------------------------------------------------
%% create look up table 
%--------------------------------------------------------------------------
tic
clc
fprintf('generating dictionaries...\n');

t1_entries  = [5:10:3000, 3100:50:5000];
t2_entries  = [1:2:350, 370:20:1000, 1100:100:3000];
t2s_entries  = [1:1:100, 105:5:200 210:50:500];


T1_entries  = repmat(t1_entries.', [1,length(t2_entries)]).';
T1_entries  = T1_entries(:);
  
T2_entries  = repmat(t2_entries.', [1,length(t1_entries)]);
T2_entries  = T2_entries(:);


t1t2_lut    = cat(2, T1_entries, T2_entries);

% remove cases where T2>T1
idx = 0;
for t = 1:length(t1t2_lut)
    if t1t2_lut(t,1) < t1t2_lut(t,2)
        idx = idx+1;
    end
end

t1t2_lut_prune = zeross( [length(t1t2_lut) - idx, 2] );


ie_lkp = reshape((ielookup.ies_mtx),[length(t1t2_lut),1]);
ie_lkp_prune = zeross( [length(t1t2_lut) - idx, 1] );
%% prune
idx = 0;
for t = 1:length(t1t2_lut)
    if t1t2_lut(t,1) >= t1t2_lut(t,2)
        idx = idx+1;
        t1t2_lut_prune(idx,:) = t1t2_lut(t,:);
        ie_lkp_prune(idx,:) = ie_lkp(t,:);
    end
end

disp(['dictionary entries: ', num2str(length(t1t2_lut_prune))])
%%

%%
%--------------------------------------------------------------------------
% dictionary fit -> in each slice, bin voxels based on b1 value
%--------------------------------------------------------------------------

length_dict = length(t1t2_lut_prune);
fprintf('fitting...\n');

estimate_pd_map = 1;     % set to 1 to estiamte PD map (makes it slower)

T1_map = zeross(N);
T2_map = zeross(N);
% T2s_map = zeross(N);

PD_map = zeross(N);     % proton density-> won't be estimated if estimate_pd_map is set to 0
IE_map = zeross(N);     % inversion efficiency

iniTime         = clock;
iniTime0        = iniTime;


% parallel computing
% delete(gcp('nocreate'))
c = parcluster('local');
total_cores = c.NumWorkers;
parpool(min(ceil(total_cores*.25), 10))

parfor slc_select = 1:N(3)
% for slc_select = 1:N(3)
    disp(num2str(slc_select))

    tic
    for t2ss = 1:length(T2s_val)
        msk_slc_t2s = msk_t2s(:,:,slc_select,t2ss);
        % msk_slc = new_mask(:,:,b);
        num_vox = sum(msk_slc_t2s(:)~=0);

        if num_vox > 0

            for b = 1:length(b1_val)
                msk_slc_b1 = msk_b1(:,:,slc_select,b);
                num_vox_b1 = sum(msk_slc_b1(:)~=0);
                msk_slc = msk_slc_t2s.*msk_slc_b1;
                % te = msk_slc + te;
                num_vox_all = sum(msk_slc(:)~=0);
                if num_vox_all > 0
                    
                    img_slc = zeross([7, num_vox_all]);
    
                    for t = 1:7
                        temp = sq(img(:,:,slc_select,t));
                        img_slc(t,:) = temp(msk_slc~=0);
                    end
                        if sum(img_slc,'all')~=0
                        ww  =1;
                        end


                        [~, t2s_idx] = min(abs(t2s_entries - T2s_val(t2ss)));
                        
                        res = dict(:,:,t2s_idx,b) * img_slc; % dot product
                        
               
                        % find T1, T2 values
                        [~, max_idx] = max(abs(res), [], 1);
        
                        max_idx_t1t2 = mod(max_idx, length_dict);
                        max_idx_t1t2(max_idx_t1t2==0) = length_dict;
        
                        res_map = t1t2_lut_prune(max_idx_t1t2,:);

                        ie_to_use = ie_lkp_prune(max_idx_t1t2,1);
        
                        if estimate_pd_map
                              [Mz_sim, Mxy_sim] = sim_MIMOSA_for7T_v2(TR, alpha_deg, esp, turbo_factor, res_map(:,1)*1e-3, res_map(:,2)*1e-3, num_reps, echo2use,TR_mte,esp_mte,TEs ,T2s_val(t2ss)*1e-3, gap_between_readouts, time2relax_at_the_end, b1_val(b), ie_to_use);
                        end
                        
        
                    t1_map = zeross(N(1:2));
                    t1_map(msk_slc==1) = res_map(:,1);
        
                    t2_map = zeross(N(1:2));
                    t2_map(msk_slc==1) = res_map(:,2);
                    

        
                    ie_map = zeross(N(1:2));
                    ie_map(msk_slc==1) = ie_to_use;
        
                    if estimate_pd_map
                        Mxy_sim_use = abs(Mxy_sim(:,:,end));
        
                        scl = zeross([num_vox_all,1]);
        
                        for idx = 1:size(Mxy_sim_use,2)
                            scl(idx) = Mxy_sim_use(:,idx) \ img_slc(:,idx);
                        end
        
                        pd_map = zeross(N(1:2));
                        pd_map(msk_slc~=0) = scl;
                        PD_map(:,:,slc_select) = PD_map(:,:,slc_select) + pd_map;
                    end
        
                    T1_map(:,:,slc_select) = T1_map(:,:,slc_select) + t1_map;
                    T2_map(:,:,slc_select) = T2_map(:,:,slc_select) + t2_map;

                    IE_map(:,:,slc_select) = IE_map(:,:,slc_select) + ie_map;
    
                end
            end
        end
    end
    toc 
end
%% 3D 
T2s_map = T2std.*mask_brain;
imagesc3d2(sq(T1_map), s(sq(T1_map(:,:,:)))/2, 1, [180 180 180], [0 4000]);colormap hot
imagesc3d2(sq(T2_map), s(sq(T1_map(:,:,:)))/2, 2, [180 180 180], [0 100]);colormap hot
imagesc3d2(sq(T2s_map), s(sq(T1_map(:,:,:)))/2, 3, [180 180 180], [0 100]);colormap hot
imagesc3d2(sq(PD_map), s(sq(T1_map(:,:,:)))/2, 4, [180 180 180], [0 max(abs(PD_map(:)))]);colormap gray
imagesc3d2(sq(IE_map), s(sq(T1_map(:,:,:)))/2, 5, [180 180 180], [0 1]);colormap gray

%% χ-separation Tool

% This tool is MATLAB-based software forseparating para- and dia-magnetic susceptibility sources (χ-separation). 
% Separating paramagnetic (e.g., iron) and diamagnetic (e.g., myelin) susceptibility sources 
% co-existing in a voxel provides the distributions of two sources that QSM does not provides. 

% χ-separation tool v1.0

% Contact E-mail: snu.list.software@gmail.com 

% Reference
% H.-G. Shin, J. Lee, Y. H. Yun, S. H. Yoo, J. Jang, S.-H. Oh, Y. Nam, S. Jung, S. Kim, F. Masaki, W. 
% Kim, H. J. Choi, J. Lee. χ-separation: Magnetic susceptibility source separation toward iron and 
% myelin mapping in the brain. Neuroimage, 2021 Oct; 240:118371.

% χ-separation tool is powered by MEDI toolbox (for BET), STI Suite (for V-SHARP), SEGUE toolbox (for SEGUE), and mritools (for ROMEO).



%%%%%%%%%%%%%%%%%%%% Necessary preparation

% Set x-separation tool directory path
% home_directory = 'E:\MATLAB\demo\chi-separation-main\chi-separation-main\Chisep_Toolbox_v1.0.0';
home_directory ='E:\MATLAB\demo\chi-separation_v2\chi-separation-main\Chisep_Toolbox_v1.1.2';
addpath(genpath(home_directory))

% Set MATLAB tool directory path 
% xiangruili/dicm2nii (https://kr.mathworks.com/matlabcentral/fileexchange/42997-xiangruili-dicm2nii)
addpath(genpath('E:\MATLAB\demo\chi-separation-main\xiangruili-dicm2nii-3fe1a27'))
% Tools for NIfTI and ANALYZE image (https://kr.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
addpath(genpath('E:\MATLAB\demo\chi-separation-main\NIfTI_20140122'))

% Download onnxconverter Add-on, and then install it.
% Deep Learning Toolbox Converter for ONNX Model Format 
% (https://kr.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format)

% Set QSM tool directory path 
% STI Suite (Version 3.0) (https://people.eecs.berkeley.edu/~chunlei.liu/software.html)
addpath(genpath('E:\MATLAB\demo\chi-separation-main\STISuite_V3.0\STISuite_V3.0'))

% MEDI toolbox (http://pre.weill.cornell.edu/mri/pages/qsm.html)
addpath(genpath('E:\MATLAB\demo\chi-separation-main\MEDI_toolbox'))

% SEGUE toolbox (https://xip.uclb.com/product/SEGUE)
addpath(genpath('E:\MATLAB\demo\chi-separation-main\SEGUE_28012021\SEGUE_28012021'))

% mritools toolbox (https://github.com/korbinian90/CompileMRI.jl/releases)
addpath(genpath('E:\MATLAB\demo\chi-separation-main\mritools_windows-2019_4.0.6'))

%% Data paramters
% B0_strength, B0_direction, CF (central frequency), TE (echo time), delta_TE, voxel_size

% B0_strength = 7;
% B0_direction = [0, 0, 1];
% CF = 123200000;
% TE = [0.00xx, 0.00xx, 0.00xx, 0.00xx, ...];
% delta_TE = 0.00xx;
% voxel_size = [dx, dy, dz];

cplx_img = img_zsssl(:,:,:,4:end);

cplx_img = sqrt(abs(cplx_img)).*exp(1i*angle(cplx_img));
mag_multi_echo = abs(cplx_img);
phs_multi_echo = angle(cplx_img);

B0_strength = 7;
B0_direction = [1, 0, 0];%  for data with QALAS head
CF = 123200000/3*7;
TE = [3e-3:7e-3:25e-3];
delta_TE  = TE(2) - TE(1);
voxel_size = [0.75 0.75 0.75];

%% Preprocessing
% Tukey windowing
tukey_fac = 0.4; % Recommendation: Siemens, GE: 0.4, Philips: 0
img_tukey = tukey_windowing(mag_multi_echo.* exp(1i*phs_multi_echo),tukey_fac);
mag_multi_echo = abs(img_tukey);
phs_multi_echo = angle(img_tukey);
% Compute single magnitude data
mag = sqrt(sum(abs(mag_multi_echo).^2,4));
[~, N_std] = Preprocessing4Phase(mag_multi_echo, phs_multi_echo);

%% R2* mapping
% Compute R2* (need multi-echo GRE magnitude)

if(use_arlo(TE))
    % Use ARLO (More than three equi-spaced TE needed)
    r2star = r2star_arlo(mag_multi_echo,TE*1000,mask_brain); % Convert TE to [ms]
else
% Use NNLS fitting (When ARLO is not an option)
    r2star = r2star_nnls(mag_multi_echo,TE*1000,mask_brain); % Convert TE to [ms]
end
%% Phase unwrapping and Echo combination
% 1. ROMEO + weighted echo averaging 
parameters.TE = TE * 1000; % Convert to ms
parameters.mag = mag_multi_echo;
parameters.mask = double(mask_brain);
parameters.calculate_B0 = false;
parameters.phase_offset_correction = 'on';
parameters.voxel_size = voxel_size;
parameters.additional_flags = '-q';%'--verbose -q -i'; % settings are pasted directly to ROMEO cmd (see https://github.com/korbinian90/ROMEO for options)
parameters.output_dir = 'romeo_tmp'; % if not set pwd() is used
mkdir(parameters.output_dir);

[unwrapped_phase, B0] = ROMEO(double(phs_multi_echo), parameters);
unwrapped_phase(isnan(unwrapped_phase))= 0;

% Weighted echo averaging
t2s_roi = 0.04; % in [s] unit
W = (TE).*exp(-(TE)/t2s_roi);
weightedSum = 0;
TE_eff = 0;
for echo = 1:size(unwrapped_phase,4)
    weightedSum = weightedSum + W(echo)*unwrapped_phase(:,:,:,echo)./sum(W);
    TE_eff = TE_eff + W(echo)*(TE(echo))./sum(W);
end

field_map = weightedSum/TE_eff*delta_TE.*mask_brain; % Tissue phase in rad
% field_map = -field_map; % If the phase rotation is in the opposite direction
%% Background field removal
% V-SHARP from STI Suite
smv_size=12;
[local_field, mask_brain_new]=V_SHARP(field_map, mask_brain,'voxelsize',voxel_size,'smvsize',smv_size);
local_field_hz = local_field / (2*pi*delta_TE); % rad to hz

%% χ-separation
% Parameters for using r2
have_r2map = true;% use T2mapping of QALAS
r2 = 1./(T2_map*1e-3);% unitHz

if ~have_r2map
    % Compute pseudo R2map (If you don't have r2map)
    nominal_value = 13;
    r2 = mask_brain;
    r2(r2~=0) = nominal_value;
end


% Compute R2'
r2prime = r2star - r2;
r2prime(r2prime<0) = 0; 

%% chi_sepnet,v1.1.2
have_r2prime = exist('r2prime','var');

if have_r2prime
    % Use Chi-sepnet-R2'
    map = r2prime;
else
    % Use Chi-sepnet-R2*
    map = r2star;
end
Dr = 114; % This parameter is different from the original paper (Dr = 137) because the network is trained on COSMOS-reconstructed maps

resgen = true; % Determine whether to use resolution generalization pipeline or to interpolate to 1 mm isotropic resolution
if resgen
    % Use the resolution generalization pipeline. Resolution of input data is retained in the resulting chi-separation maps
    [x_para, x_dia, x_tot] = chi_sepnet_general_new_wResolGen(home_directory, local_field_hz, map, mask_brain_new, Dr, B0_direction, CF, voxel_size, have_r2prime);
else
    % Interpolate the input maps to 1 mm isotropic resolution. Output resolution is also 1 mm isotropic.
    [x_para, x_dia, x_tot] = chi_sepnet_general(home_directory, local_field_hz, map, mask_brain_new, Dr, B0_direction, CF, voxel_size, have_r2prime);
end
imagesc3d2( permute(x_para,[2 3 1]), s(permute(x_para,[2 3 1]))/2+[40 0 10], 2, [-90 0 0], [0 0.1])
imagesc3d2( x_dia, s(x_dia)/2+[26 0 10], 3, [0,0,0], [0 0.1])
imagesc3d2( x_tot, s(x_tot)/2+[26 0 10], 4, [0,0,0], [-0.1 0.1])
%% save
save('mapping_zsssl_80000_R3_dict_v1_b1Cor.mat','T1_map','T2_map','T2s_map','PD_map','IE_map','x_para','x_dia','x_tot')






