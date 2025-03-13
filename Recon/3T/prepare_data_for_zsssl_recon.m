
%--------------------------------------------------------------------------
%% Read twix data and sort kspace based on seq file
%--------------------------------------------------------------------------

clear; clc;

addpath(genpath('utils'));


flag.save_kspace = 1;
flag.save_nii = 1;

data_file_path='/rawdata/meas_MID00017_FID51769_MIMOSA_1iso_R11_d1_cplm_v2.dat';

[p,n,e] = fileparts(data_file_path);
basic_file_path=fullfile(p,n);

twix_obj = mapVBVD(data_file_path);
data_unsorted = twix_obj{end}.image.unsorted();
[adc_len,ncoil,readouts]=size(data_unsorted);


% Read params from seq file
pulseq_file_path = [pwd '/MIMOSA_1iso_R11' '.seq'];
seq=mr.Sequence();
seq.read(pulseq_file_path);


N = seq.getDefinition('Matrix');
nTR = seq.getDefinition('nTR');
nETL = seq.getDefinition('nETL');
%os_factor = seq.getDefinition('os_factor');
os_factor=1;
traj_y = seq.getDefinition('traj_y');
traj_z = seq.getDefinition('traj_z');
step_size = 3 * nETL;
%#######
TEs = seq.getDefinition('TES_mte');
nechoes = seq.getDefinition('num_echoes');
esp_mte = seq.getDefinition('esp_mte');
TR_mte = seq.getDefinition('TR_mte');
%###########
nacq = 3+6;
%% prescan
ny_lr = 32;
nz_lr = 32;
k_ref = reshape(permute(data_unsorted(:,:,1:ny_lr*nz_lr),[1,3,2]),N(1),ny_lr,nz_lr,ncoil);
jimg2(rsos(ifft3call(k_ref),4));
save('ref.mat','k_ref');
data_unsorted_data = data_unsorted(:,:,ny_lr*nz_lr+1:end);
clear data_unsorted
%% Pre-allocate
img4d_data = zeros([N(1)*os_factor N(2) N(3) 3+nechoes]);
kspace = zeros([N(1)*os_factor N(2) N(3)  ncoil 3+nechoes]);
mask_traj = zeros([N(2) N(3) nETL 3+nechoes]);

ind13 = [];
indmte = [];
for t = 0:nTR-1
    c1_start =  (3 + nechoes)*t*nETL+1;
    c3_end = (3 + nechoes)*t*nETL + 3*nETL;
    c4_start = c3_end+1;
    c9_end = (3 + nechoes)*(t+1)*nETL;
    ind13 = [ind13, c1_start:c3_end];
    indmte = [indmte, c4_start:c9_end];
end
data_c1_to_c3 = data_unsorted_data(:,:,ind13(:));% extract data of each contrast
data_c4_to_c9 = data_unsorted_data(:,:,indmte(:));% extract data of each contrast
readout_c4 = size(data_c1_to_c3,3);

for contrast = 1:3
    indices = [];
    for segment_start = (1+nETL*(contrast-1)):step_size:readout_c4
        segment_end = min(segment_start + nETL - 1, readout_c4);
        indices = [indices, segment_start:segment_end];
    end

    data = data_c1_to_c3(:,:,indices(:));% extract data of each contrast
    kspace_contrast = zeros([N(1)*os_factor N(2) N(3) ncoil]);

    ii = 1;
    for rr= 1:nTR
        for ee = 1:nETL
            ky = traj_y((contrast-1)*nTR*nETL+ee+(rr-1)*nETL);
            kz = traj_z((contrast-1)*nTR*nETL+ee+(rr-1)*nETL);
            if sum(kspace_contrast(:,ky,kz,:),'all')==0
                kspace_contrast(:,ky,kz,:) = data(:,:,ii);% To avoid overwriting
                mask_traj(ky,kz,ee,contrast) = 1;
            else
                disp(['Skip ky=' num2str(ky) ',kz=' num2str(kz)])
            end
            ii = ii+1;
             
        end
    end

    im = fft3c2(kspace_contrast);
    im3D = abs(sum(im.*conj(im),ndims(im))).^(1/2);
    img4d_data(:,:,:,contrast) = im3D;
    kspace(:,:,:,:,contrast) = kspace_contrast;
end

%%
for contrast = 4:nechoes+3
    ind_acq4 = [];
    for i = 1:nTR
        ind_acq4 = [ind_acq4, (contrast-3)+nETL*6*(i-1):6:nETL*6*i];
    end
    data = data_c4_to_c9(:,:,ind_acq4(:));
    kspace_contrast = zeros([N(1)*os_factor N(2) N(3) ncoil]);
     % for cplm of out-center
     ii = 1;
    for nt = 1:nTR
        for yy=1:nETL
            index = (contrast-1)*nTR*nETL+(nt-1)*nETL+yy;
            index_oc =  (contrast-1)*nTR*nETL+(nt)*nETL - yy +1;
            ky = traj_y(index_oc);
            kz = traj_z(index_oc);
            if sum(kspace_contrast(:,ky,kz,:),'all')==0
                kspace_contrast(:,ky,kz,:) = data(:,:,ii);% To avoid overwriting
                mask_traj(ky,kz,ee,contrast) = 1;
            else
                disp(['Skip ky=' num2str(ky) ',kz=' num2str(kz)])
            end
            ii = ii+1;
        end
    end

    im = fft3c2(kspace_contrast);
    im3D = abs(sum(im.*conj(im),ndims(im))).^(1/2);

    img4d_data(:,:,:,contrast) = im3D;
    kspace(:,:,:,:,contrast) = kspace_contrast;
end

%%
clearvars -except kspace data_file_path

%--------------------------------------------------------------------------
%% patref scan
%--------------------------------------------------------------------------
   
load('ref.mat');
ref = flip(flip(k_ref, 2), 3);
% mosaic(rs os(ref(1+end/2,:,:,:,1),4),1,1,10,'',[0,3e-3]), setGcf(.5)

img_ref = ifft3call(ref);

imagesc3d2(rsos(img_ref,4), s(img_ref)/2, 1, [0,0,0], [-0,3e-3]), setGcf(.5)


%--------------------------------------------------------------------------
%% coil compression
%--------------------------------------------------------------------------

num_chan = 32;  % num channels to compress to

[ref_svd, cmp_mtx] = svd_compress3d(ref, num_chan, 1);  

rmse(rsos(ref_svd,4), rsos(ref,4))

N = size(kspace(:,:,:,1,1));
num_eco = size(kspace,5);

kspace_svd = zeross([N,num_chan,num_eco]);

for t = 1:size(kspace,5)
    kspace_svd(:,:,:,:,t) = svd_apply3d(kspace(:,:,:,:,t), cmp_mtx);
end

rmse(rsos(kspace_svd,4), rsos(kspace,4))
clear kspace

[p,n,e] = fileparts(data_file_path);
mat_file_path = fullfile(p,[n,'_svd_skip','.mat']);
save(mat_file_path,"kspace_svd","-v7.3")
%--------------------------------------------------------------------------
%% interpolate patref by zero padding to the high res matrix size
%--------------------------------------------------------------------------

size_data = size(kspace_svd(:,:,:,1,1));
size_patref = size(ref_svd(:,:,:,1,1));
% pad ref to be the same size
patref_pad = padarray( ref_svd, [size_data-size_patref, 0, 0, 0]/2 );

img_patref_pad = ifft3c(patref_pad);

imagesc3d2( rsos(img_patref_pad,4), s(img_patref_pad)/2, 10, [0,0,0], [0,2e-4])
%--------------------------------------------------------------------------
%% calculate sens map using ESPIRiT: parfor
%--------------------------------------------------------------------------

num_acs = min(size_patref);
kernel_size = [6,6];
eigen_thresh = 0.7;

receive = zeross(size(kspace_svd(:,:,:,:,1)));


delete(gcp('nocreate'))% why shut up parallel pool?
c = parcluster('local');    

total_cores = c.NumWorkers;  
parpool(ceil(total_cores/4))

    
tic
parfor slc_select = 1:s(img_patref_pad,1)     
    disp(num2str(slc_select))
    
    [maps, weights] = ecalib_soft( fft2c( sq(img_patref_pad(slc_select,:,:,:)) ), num_acs, kernel_size, eigen_thresh );

    receive(slc_select,:,:,:) = permute(dot_mult(maps, weights >= eigen_thresh ), [1,2,4,3]);
end 
toc
receive = abs(receive) .* exp(1i * angle( receive .* repmat(conj(receive(:,:,:,1)), [1,1,1,num_chan]) ));
delete(gcp('nocreate'))
 
%% generate data for reconstruction
mask_all = kspace_svd ~= 0;
[nx, ny, nz, nc, ne] = size(kspace_svd);

tmp = fftshift(ifft(ifftshift(kspace_svd),[],1))* sqrt(nx);   % send kx into image domain

kMask = squeeze(mask_all(slice_idx,:,:,:,:));

% check data to remove background slices without information
im3d = rsos(ifft3call(sq(kspace_svd(:,:,:,:,1))),4);
imagesc3d2(sq(im3d(:,:,:,1)), s(sq(im3d(:,:,:,1)))/2, 1, [0,0,0], [-0,3e-4]), setGcf(.5)
% generate data slice by slice 
if ~exist('zsssl_recon_3T/data', 'dir')
    mkdir('zsssl_recon_3T/data'); % ´´½¨ÎÄ¼þ¼Ð
end
cd('zsssl_recon_3T/data/')
for ss =53:216% discard non-brain slices
    kspace =  squeeze(tmp(ss,:,:,:,:));
    sens_maps = squeeze(receive(ss,:,:,:));
    mask = squeeze(mask_all(1,:,:,1,:)); % nx, ny, nz, nc, ncontrast
    filename = sprintf('mimosa_slc_%03d', ss);
    save([filename '.mat'],'kspace','sens_maps','mask')
end