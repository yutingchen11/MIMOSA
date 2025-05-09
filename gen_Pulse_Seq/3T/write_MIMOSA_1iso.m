% MIMOSA sequence
% Feb. 21 2025, Yuting Chen
% ychen156@mgh.harvard.edu

clc; clear all; close all;
%--------------------------------------------------------------------------
% Define high-level parameters
%--------------------------------------------------------------------------
sys = mr.opts('MaxGrad', 32, 'GradUnit', 'mT/m', ...
              'MaxSlew', 160, 'SlewUnit', 'T/m/s', ...
              'rfDeadTime', 100e-6, ...
              'rfRingdownTime', 60e-6, ...     
              'adcDeadTime', 40e-6, ...         
              'adcRasterTime', 2e-6, ...        
              'gradRasterTime', 20e-6, ...     
              'blockDurationRaster', 20e-6, ... 
              'B0', 3.0);    
seq=mr.Sequence(sys);           % Create a new sequence object
%%
Acc = 11.75;% acceleration rate, option: 3.26, 6.47, 11.75

if Acc == 3.26
    traj = readmatrix('csv_imports/cplm/fov_224x192_msize_224x192_tl_127_ncal_0_spiral_12_acc_1.61x1.61_nTR936_nph9.txt');
elseif Acc == 6.47
    traj = readmatrix('csv_imports/cplm/fov_224x192_msize_224x192_tl_127_ncal_0_spiral_12_acc_2.25x2.25_nTR477_nph9.txt');
elseif Acc == 11.75
    traj = readmatrix('csv_imports/cplm/fov_224x192_msize_224x192_tl_127_ncal_0_spiral_12_acc_3.03x3.03_nTR261_nph9.txt');
end

traj = traj(:,[3,2,1]); 
nETL = 127;
nacq = 9;
nTR = size(traj,1)/nETL/nacq;


% Create sequence object and define FOV, resolution, and other high-level parameters
fov=[240 224 192]*1e-3;         % Define FOV and resolution  1 iso
N = [240 224 192];

Nx = N(1);
Ny = N(2);
Nz = N(3);

bandwidth = 340;                % Hz/pixel
flip_angle = 4;                 % degrees
rfSpoilingInc=117;              % RF spoiling increment
%Tread = 1 / bandwidth;          % readout duration in sec
dwell = 12e-6;
Tread = dwell*Nx;
os_factor = 1;                  % readout oversampling amount

%--------------------------------------------------------------------------
% Prepare sequence blocks
%--------------------------------------------------------------------------

deltak = 1 ./ fov;
gx = mr.makeTrapezoid('x',sys,'Amplitude',Nx*deltak(1)/Tread,'FlatTime',ceil(Tread/sys.gradRasterTime)*sys.gradRasterTime ,'system',sys);    % readout gradient
gxPre = mr.makeTrapezoid('x',sys,'Area',-gx.area/2);        % Gx prewinder
gxSpoil = mr.makeTrapezoid('x',sys,'Area',gx.area);         % Gx spoiler

% Make trapezoids for inner loop to save computation

gyPre = mr.makeTrapezoid('y','Area',deltak(2)*(Ny/2),'Duration',mr.calcDuration(gxPre), 'system',sys);
gzPre = mr.makeTrapezoid('z','Area',deltak(3)*(Nz/2),'Duration',mr.calcDuration(gxPre), 'system',sys);

gyReph = mr.makeTrapezoid('y','Area',deltak(2)*(Ny/2),'Duration',mr.calcDuration(gxSpoil), 'system',sys);
gzReph = mr.makeTrapezoid('z','Area',deltak(3)*(Nz/2),'Duration',mr.calcDuration(gxSpoil), 'system',sys);

stepsY=((traj(:,2)-1)-Ny/2)/Ny*2;
stepsZ=((traj(:,3)-1)-Nz/2)/Nz*2;

% Create non-selective pulse
rf = mr.makeBlockPulse(flip_angle*pi/180, sys, 'Duration', 0.1e-3);

% Spoilers after t2prep and IR prep
gslSp_t2prep = mr.makeTrapezoid('z','Amplitude',-42.58*8*1e3,'Risetime',0.84e-3,'Duration',0.84e-3+8e-3+0.84e-3,'system',sys); %Amplitude in Hz/m
gslSp_IRprep = mr.makeTrapezoid('z','Amplitude',-42.58*8*1e3,'Risetime',1e-3,'Duration',1e-3+8e-3+1e-3,'system',sys); %Amplitude in Hz/m

% Analog to digital convertersys
adc = mr.makeAdc(Nx * os_factor,'Duration',Tread,'Delay',gx.riseTime,'system',sys);

% Prep pulses
% T2 prep and IR prep pulse are imported from external txt files
% txt file is in mag and phase, while mr.makeArbitraryRf assumes real and imag
t2prep = readmatrix('csv_imports/T2prep.txt');
Re = t2prep(:,1) .* cos(t2prep(:,2));
Im = t2prep(:,1) .* sin(t2prep(:,2));
t2prep_pulse = mr.makeArbitraryRf((Re+Im*1i).', 380.4*pi/180, 'system',sys, 'dwell', 1e-6);

text = readmatrix('csv_imports/rf90.txt');
Re = text(:,1) .* cos(text(:,2));
Im = text(:,1) .* sin(text(:,2));
rf90 = mr.makeArbitraryRf((Re+Im*1i).', pi/2, 'system',sys, 'dwell', 1e-6);
rf90_180PhaseOffset = mr.makeArbitraryRf((Re+Im*1i).', pi/2, 'system',sys, 'dwell', 1e-6, 'PhaseOffset',-180*pi/180);

IRprep = readmatrix('csv_imports/invpulse.txt');
Re = IRprep(:,1) .* cos(IRprep(:,2));
Im = IRprep(:,1) .* sin(IRprep(:,2));
IRprep_pulse = mr.makeArbitraryRf((Re+Im*1i).', 1500*pi/180, 'system',sys, 'dwell', 1e-5);


% setup paramater for MGRE module
esp_mte = 4.3;
TEs = [2.7:esp_mte:25] * 1e-3;%
TR_mte = 27.5e-3;
nechoes = length(TEs);
gxFlyBack = mr.makeTrapezoid('x','Area',-gx.area,'system',sys);

delayTE_mte = zeros(nechoes,1);
delayTE_mte(1) = ceil((TEs(1) - (mr.calcDuration(rf) - mr.calcRfCenter(rf)-rf.delay) - mr.calcDuration(gxPre) - mr.calcDuration(gx)/2)/seq.gradRasterTime)*seq.gradRasterTime;
for c = 2:nechoes
    delayTE_mte(c) = ceil(( TEs(c) - TEs(c-1) - mr.calcDuration(gx)  - mr.calcDuration(gxFlyBack))/seq.gradRasterTime)*seq.gradRasterTime;
    if delayTE_mte(c) < 0
        disp(['echo ', num2str(c), ' cannot be fit'])
    else
        disp(['echo ', num2str(c), ' delay ', num2str(1e3*delayTE_mte(c)), ' ms'])
    end
end

delayTR_mte = ceil((TR_mte - mr.calcDuration(rf) - mr.calcDuration(gxPre)  ...
    - mr.calcDuration(gxSpoil) - sum(delayTE_mte) - mr.calcDuration(gx)*length(TEs)...
    - mr.calcDuration(gxFlyBack)*(nechoes-1))/seq.gradRasterTime)*seq.gradRasterTime;
disp(['delay TR: ', num2str(delayTR_mte*1e3), ' ms'])
dTR=mr.makeDelay(delayTR_mte);


%-------------------------------------------------------------------------
% Adjust sequence timings
%--------------------------------------------------------------------------

esp = 5.8e-3;
gap_between_readouts = 900e-3;

delay_1_t2prep  =   11e-3 + 80e-6;
delay_2_t2prep  =   25e-3;        
delay_3_t2prep  =   14e-3 - 80e-6;
delay_IRprep    =   100e-3 - mr.calcDuration(IRprep_pulse)/2;         % gap between end of inversion and start of readout#2 
delay_TE        =   0;
delay_TRinner   =   esp - (mr.calcDuration(rf) + delay_TE + mr.calcDuration(gxPre)+mr.calcDuration(gx)+mr.calcDuration(gxSpoil));         
delay_TRouter = 1e-3;
delT_M3_M4      =   gap_between_readouts - esp*nETL - mr.calcDuration(IRprep_pulse) - delay_IRprep;     % between end of readout#1 and start of inversion
delT_M3_M4      =   delT_M3_M4 - 0.22e-3;
delT_M13_2end   =   53.5e-3;

%% calibaration scan 
Ny_ref = 32;
Nz_ref = 32;

TE_ref = 5e-3;
TR_ref = 12e-3;

rf_acs = rf;

delayTE_ref = ceil( (TE_ref - mr.calcDuration(rf_acs) + mr.calcRfCenter(rf_acs) + rf_acs.delay - mr.calcDuration(gxPre)  ...
    - mr.calcDuration(gx)/2)/seq.gradRasterTime)*seq.gradRasterTime;

if delayTE_ref < 0
    disp('error: acs delay TE is negative')    
else
    disp(['acs delay TE: ', num2str(1e3*delayTE_ref), ' ms'])
end


delayTR_ref = ceil((TR_ref - mr.calcDuration(rf_acs) - delayTE_ref- mr.calcDuration(gxPre) ...
    - mr.calcDuration(gx) - mr.calcDuration(gxSpoil))/seq.gradRasterTime)*seq.gradRasterTime;

if delayTR_ref < 0
    disp('error: acs delay TR is negative')    
else
    disp(['acs delay TR: ', num2str(delayTR_ref*1e3), ' ms'])
end


%--------------------------------------------------------------------------
%  dummies
%--------------------------------------------------------------------------
Ndummy_acs = 50;
areaY = ((0:Ny-1)-Ny/2)*deltak(2);
areaZ = ((0:Nz-1)-Nz/2)*deltak(3);
gyPre_acs = mr.makeTrapezoid('y','Area',areaY(floor(Ny/2)),'Duration',mr.calcDuration(gxPre), 'system',sys);
gyReph_acs = mr.makeTrapezoid('y','Area',-areaY(floor(Ny/2)),'Duration',mr.calcDuration(gxPre), 'system',sys);

% Drive magnetization to steady state
rf_phase=0;
rf_inc=0;
for iY = 1:Ndummy_acs
    % RF spoiling
    rf_acs.phaseOffset=rf_phase/180*pi;
    rf_inc=mod(rf_inc+rfSpoilingInc, 360.0);
    rf_phase=mod(rf_phase+rf_inc, 360.0);       %increment RF phase
    seq.addBlock(rf_acs);


    % Gradients    
    seq.addBlock(gxPre,gyPre_acs);                  % add Gx pre-winder, go to desired ky
    seq.addBlock(mr.makeDelay(delayTE_ref));    % add delay needed before the start of readout

    seq.addBlock(gx);                           % add readout Gx

    seq.addBlock(gyReph_acs,gxSpoil);               % add Gx spoiler, and go back to DC in ky
    seq.addBlock(mr.makeDelay(delayTR_ref))     % add delay to the end of TR
end


temp = 1:Ny;
iY_ref_indices = temp(1+end/2-Ny_ref/2:end/2+Ny_ref/2);

temp = 1:Nz;
iZ_ref_indices = temp(1+end/2-Nz_ref/2:end/2+Nz_ref/2);

% Make trapezoids for inner loop to save computation
for iY = iY_ref_indices
    gyPre_acs(iY) = mr.makeTrapezoid('y','Area',areaY(iY),'Duration',mr.calcDuration(gxPre), 'system',sys);
    gyReph_acs(iY) = mr.makeTrapezoid('y','Area',-areaY(iY),'Duration',mr.calcDuration(gxPre), 'system',sys);
end

%--------------------------------------------------------------------------
% ref data:
%--------------------------------------------------------------------------

mask_ref = zeros([Ny,Nz]);

for iZ = iZ_ref_indices
    % Gz blips to go desired kz, and to come back to DC in kz
    gzPre_acs = mr.makeTrapezoid('z','Area',areaZ(iZ),'Duration',mr.calcDuration(gxPre), 'system',sys);         
    gzReph_acs = mr.makeTrapezoid('z','Area',-areaZ(iZ),'Duration',mr.calcDuration(gxPre), 'system',sys);

    for iY = iY_ref_indices
        mask_ref(iY,iZ) = 1;

        % RF spoiling
        rf_acs.phaseOffset=rf_phase/180*pi;
        adc.phaseOffset=rf_phase/180*pi;
        rf_inc=mod(rf_inc+rfSpoilingInc, 360.0);
        rf_phase=mod(rf_phase+rf_inc, 360.0);       %increment RF phase

        % Excitation
        seq.addBlock(rf_acs);

        % Encoding
        seq.addBlock(gxPre,gyPre_acs(iY),gzPre_acs);        % Gz, Gy blips, Gx pre-winder
        seq.addBlock(mr.makeDelay(delayTE_ref));    % delay until readout

        seq.addBlock(gx,adc);                       % readout

        seq.addBlock(gyReph_acs(iY),gzReph_acs,gxSpoil);% -Gz, -Gy blips, Gx spoiler
        seq.addBlock(mr.makeDelay(delayTR_ref))     % wait until end of TR
    end
end

mosaic(mask_ref, 1, 1, 20), 



%% Dummy scan of MIMOSA
nDummies = 1;
useAdc = 0;% NO in dummy scanning ADC
mask_traj = zeros([N(2) N(3) nacq]);% to check mask
for iZ = 1:nDummies
    rf_phase=0;
    rf_inc=0;

    % T2 prep pulse
    seq.addBlock(rf90,mr.makeDelay(delay_1_t2prep));
    seq.addBlock(t2prep_pulse,mr.makeDelay(delay_2_t2prep));
    seq.addBlock(t2prep_pulse,mr.makeDelay(delay_2_t2prep));
    seq.addBlock(t2prep_pulse,mr.makeDelay(delay_2_t2prep));
    seq.addBlock(t2prep_pulse,mr.makeDelay(delay_3_t2prep));
    seq.addBlock(rf90_180PhaseOffset);
    seq.addBlock(gslSp_t2prep);

    % FLASH readout 1
    ind_acq = 1;
    [rf_phase, rf_inc, mask_traj] = addAcq(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,nacq,ind_acq,traj,nTR,mask_traj,useAdc);
    seq.addBlock(mr.makeDelay(delT_M3_M4));

    % IR prep
    seq.addBlock(IRprep_pulse);
    seq.addBlock(gslSp_IRprep,mr.makeDelay(delay_IRprep));

    % FLASH readout 2
    ind_acq = 2;
    [rf_phase, rf_inc, mask_traj] = addAcq(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,nacq,ind_acq,traj,nTR,mask_traj,useAdc);
    seq.addBlock(mr.makeDelay(delay_TRouter))

    % FLASH readout 3
    ind_acq = 3;
    [rf_phase, rf_inc, mask_traj] = addAcq(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,nacq,ind_acq,traj,nTR,mask_traj,useAdc);
    seq.addBlock(mr.makeDelay(delay_TRouter));
    
    % MGRE Module
    ind_acq = 4;
    [rf_phase, rf_inc, mask_traj] = addAcq_mte(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,TEs, gxFlyBack,delayTE_mte, delayTR_mte,nacq,ind_acq,traj,nTR,mask_traj,useAdc);
    seq.addBlock(mr.makeDelay(delT_M13_2end));

end

% --------------------------------------------------------------------------
% Build MIMOSA sequence
%--------------------------------------------------------------------------
mask_traj = zeros([N(2) N(3) nacq]);% to check mask
useAdc = 1; % use ADC
for iZ = 1:nTR
    rf_phase=0;
    rf_inc=0;

    % T2 prep pulse
    seq.addBlock(rf90,mr.makeDelay(delay_1_t2prep));
    seq.addBlock(t2prep_pulse,mr.makeDelay(delay_2_t2prep));
    seq.addBlock(t2prep_pulse,mr.makeDelay(delay_2_t2prep));
    seq.addBlock(t2prep_pulse,mr.makeDelay(delay_2_t2prep));
    seq.addBlock(t2prep_pulse,mr.makeDelay(delay_3_t2prep));
    seq.addBlock(rf90_180PhaseOffset);
    seq.addBlock(gslSp_t2prep);

    % % FLASH readout 1
    ind_acq = 1;
    [rf_phase, rf_inc, mask_traj] = addAcq(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,nacq,ind_acq,traj,nTR,mask_traj,useAdc);
    seq.addBlock(mr.makeDelay(delT_M3_M4));

    % IR prep
    seq.addBlock(IRprep_pulse);
    seq.addBlock(gslSp_IRprep,mr.makeDelay(delay_IRprep));

    % % FLASH readout 2
    ind_acq = 2;
    [rf_phase, rf_inc, mask_traj] = addAcq(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,nacq,ind_acq,traj,nTR,mask_traj,useAdc);
    seq.addBlock(mr.makeDelay(delay_TRouter))

    % % FLASH readout 3
    ind_acq = 3;
    [rf_phase, rf_inc, mask_traj] = addAcq(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,nacq,ind_acq,traj,nTR,mask_traj,useAdc);
    seq.addBlock(mr.makeDelay(delay_TRouter));

    % MGRE Module
    ind_acq = 4;
    [rf_phase, rf_inc, mask_traj] = addAcq_mte(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,TEs, gxFlyBack,delayTE_mte, delayTR_mte,nacq,ind_acq,traj,nTR,mask_traj,useAdc);
    seq.addBlock(mr.makeDelay(delT_M13_2end));

end

%--------------------------------------------------------------------------
% Check timing and write sequence
%--------------------------------------------------------------------------

% check whether the timing of the sequence is correct
[ok, error_report]=seq.checkTiming;

if (ok)
    fprintf('Timing check passed successfully\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end


% Set definitions
seq.setDefinition('FOV', fov);
seq.setDefinition('Matrix', N);
seq.setDefinition('nETL', nETL);
seq.setDefinition('nTR', nTR);
seq.setDefinition('traj_y', traj(:,2));
seq.setDefinition('traj_z', traj(:,3));
seq.setDefinition('os_factor', os_factor);
seq.setDefinition('TES_mte', TEs);
seq.setDefinition('num_echoes', nechoes);
seq.setDefinition('esp_mte', esp_mte);
seq.setDefinition('TR_mte', TR_mte);

% plot
seq.plot('TimeRange',[9.8 10],'timeDisp','ms');

% Write to pulseq file
filename = strrep(mfilename, 'write', '');


%% check traj
mask_acq = mask_traj~=0;

mosaic(mask_traj,2,5,5,'',[0 1])
 
disp(['Undersampling Rate = :',num2str(1./mean(mask_acq(:))) '; Rnet = ',num2str(1./mean(mask_acq(:))/(4/pi))])
disp(['TA = ',num2str(6.03*nTR/60)])
%% save

seq.write(['MIMOSA_1iso_R',num2ste(round(Acc,0)),'_calib.seq']);


%--------------------------------------------------------------------------
%% Functions
%--------------------------------------------------------------------------



function [rf_phase, rf_inc, mask_traj] = addAcq(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,nacq,ind_acq,traj,nTR,mask_traj,useAdc)
for iY = 1:nETL

    % Calculate index for ky-kz look up table
    index = iY+ (ind_acq-1)*nTR*nETL + (iZ-1)*nETL;% $ cplm

    mask_traj(traj(index,2),traj(index,3),ind_acq) = 1;

    % RF spoiling
    rf.phaseOffset=rf_phase/180*pi;
    adc.phaseOffset=rf_phase/180*pi;
    rf_inc=mod(rf_inc+rfSpoilingInc, 360.0);
    rf_phase=mod(rf_phase+rf_inc, 360.0);       %increment RF phase

    % Excitation
    seq.addBlock(rf);

    % Encoding
    seq.addBlock(gxPre, ...
        mr.scaleGrad(gyPre,-stepsY(index)), ...
        mr.scaleGrad(gzPre,-stepsZ(index)));    % Gz, Gy blips, Gx pre-winder

    if useAdc
        seq.addBlock(gx, adc);                      % Gx readout
    else
        seq.addBlock(gx);                           % Gx readout
    end
    
    seq.addBlock(gxSpoil, ...
        mr.scaleGrad(gyReph,stepsY(index)), ...
        mr.scaleGrad(gzReph,stepsZ(index)));    % -Gz, -Gy blips, Gx spoiler

    seq.addBlock(mr.makeDelay(delay_TRinner));  % wait until desired echo spacing
end
end

function [rf_phase, rf_inc, mask_traj] = addAcq_mte(seq, nETL, iZ, rf, adc, rfSpoilingInc, rf_phase, rf_inc, stepsZ, stepsY, gxPre, gx, gxSpoil, delay_TE, delay_TRinner, gyPre,gyReph,gzPre,gzReph,TEs, gxFlyBack,delayTE_mte, delayTR_mte,nacq,ind_acq,traj,nTR,mask_traj,useAdc)
for iY = 1:nETL


    % RF spoiling
    rf.phaseOffset=rf_phase/180*pi;
    adc.phaseOffset=rf_phase/180*pi;
    rf_inc=mod(rf_inc+rfSpoilingInc, 360.0);
    rf_phase=mod(rf_phase+rf_inc, 360.0);       %increment RF phase

    % Excitation
    seq.addBlock(rf);
    
    % multiecho Encoding
    for c=1:length(TEs) % loop over TEs

        %###########
        % Calculate index for ky-kz look up table
        ind_acq_me = ind_acq + c - 1 ;
        % outcenter-ordering
        index = (ind_acq_me-1)*nTR*nETL - iY + 1 + iZ*nETL;
        mask_traj(traj(index,2),traj(index,3),ind_acq_me) = 1;

        if c==1 
            seq.addBlock(gxPre, ...
                mr.scaleGrad(gyPre,-stepsY(index)), ...
                mr.scaleGrad(gzPre,-stepsZ(index)));    % Gz, Gy blips, Gx pre-winder
            seq.addBlock(mr.makeDelay(delayTE_mte(c)));
        else
            seq.addBlock(mr.makeDelay(delayTE_mte(c)));
            seq.addBlock(gxFlyBack, ...
                mr.scaleGrad(gyReph,-(stepsY(index) - stepsY(ind_pre))), ...
                mr.scaleGrad(gzReph,-(stepsZ(index)- stepsZ(ind_pre))));    % Gz, Gy blips, Gx pre-winder
        end

        if useAdc
            seq.addBlock(gx, adc);                      % Gx readout
        else
            seq.addBlock(gx);                           % Gx readout
        end
        
        ind_pre = index;
    end

    seq.addBlock(gxSpoil, ...
        mr.scaleGrad(gyReph,stepsY(index)), ...
        mr.scaleGrad(gzReph,stepsZ(index)));    % -Gz, -Gy blips, Gx spoiler

    seq.addBlock(mr.makeDelay(delayTR_mte));  % wait until desired echo spacing
end
end