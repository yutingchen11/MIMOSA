function [Mz_mtx, Mxy_mtx] = sim_qalas_4acq_pd_b1_eff_T2_outc_mge_t2s_IE_d1_v2(TR, alpha_deg, esp, turbo_factor, t1_vals, t2_vals, num_reps, echo2use, TR_mte,esp_mte,TEs, t2s_vals, gap_between_readouts, time2relax_at_the_end, b1, inv_eff,pd_map )
%
% TR parameter : excludes dead time at the end of each TR ( total_TR is TR + time2relax_at_the_end )
% v2: considering T2* decay during FLASH readout

num_voxels = length(t1_vals);

if nargin < 17
    M0 = ones(num_voxels,1);
else
    M0 = pd_map(:);
end

if nargin < 16
    inv_eff = 1;
end

if nargin < 15
    b1 = 1;
end

if nargin < 14
    time2relax_at_the_end = 0;
end

if nargin < 13
    gap_between_readouts = 900e-3;
end


relax_T2 = @(Mz, TE_T2prep, T2)  Mz .* exp(-TE_T2prep ./ T2);

inverse_eff = @(M0, IE)  -1.*M0 .* IE;

relax_T2_b1 = @(Mz, TE_T2prep, T2, b1, T1)  Mz .* ( sin(b1 .* pi/2).^2 .* exp(-TE_T2prep ./ T2) + cos(b1 .* pi/2).^2 .* exp(-TE_T2prep ./ T1) );

relax_T1 = @(M0, Mz, delta_t, T1)  M0 - (M0 - Mz) .* exp(-delta_t ./ T1);

relax_T2s = @(Mxy, TE1, T2s)  Mxy .* exp(-TE1 ./ T2s);

etl = turbo_factor * esp;

% nechoes = length(TEs);
tf = turbo_factor;       % Turbo factor, or the number of echoes.
alpha = b1 .* alpha_deg * pi/180;

TE_flash = 2.29e-3;
% timings are based on: https://jcmr-online.biomedcentral.com/track/pdf/10.1186/s12968-014-0102-0.pdf

% BB: hack
delT_M1_M2 = 109.7e-3;                % duration of T2 prep
% delT_M1_M2 = 110e-3;                % duration of T2 prep

% timings below are hard coded:
% delT_M0_M1 = 900e-3 - etl - 110e-3;
% delT_M0_M1 = 900e-3 - etl - delT_M1_M2;
% delT_M0_M1 = gap_between_readouts - etl - delT_M1_M2;
% ##########################test###############
delT_M0_M1 = 0;


delT_M2_M3 = etl;                   % duration of readout#1


% delT_M2_M6 = 900e-3;                % between two readouts
delT_M2_M6 = gap_between_readouts;                % between two readouts
delT_M4_M5 = 12.8e-3;               % inversion pulse


% delT_M5_M6 = 100e-3 - 6.4e-3;       % gap between end of inversion and start of readout#2
delT_M5_M6 = 100e-3 - 6.45e-3;       % gap between end of inversion and start of readout#2

delT_M3_M4 = delT_M2_M6 - delT_M2_M3 - delT_M4_M5 - delT_M5_M6;     % between end of readout#1 and start of inversion


delT_M6_M7 = etl;                   % duration of readout#2
% delT_M7_M8 = 900e-3 - etl;          % from end of readout#2 to  begin of readout#3
% delT_M7_M8 = gap_between_readouts - etl;          % from end of readout#2 to  begin of readout#3
delT_M7_M8 = 1e-3;
delT_M8_M9 = etl;                   % duration of readout#3

% delT_M9_M10 = 900e-3 - etl;         % from end of readout#3 to  begin of readout#4
% delT_M9_M10 = gap_between_readouts - etl;         % from end of readout#3 to  begin of readout#4
delT_M9_M10 = 1e-3;
delT_M10_M11 = etl;                 % duration of readout#4

% delT_M11_M12 = 900e-3 - etl;        % from end of readout#4 to  begin of readout#5
delT_M11_M12 = gap_between_readouts - etl;        % from end of readout#4 to  begin of readout#5
delT_M12_M13 = etl;                 % duration of readout#5
% ######
delT_M12_M13_mte = TR_mte*turbo_factor;

total_event_duration = delT_M0_M1 + delT_M1_M2 + delT_M2_M3 + delT_M3_M4 + delT_M4_M5 + delT_M5_M6 + delT_M6_M7 + delT_M7_M8 + delT_M8_M9 + delT_M9_M10  + delT_M12_M13_mte;

disp(['total event duration: ', num2str(total_event_duration), ' sec'])

%##################test##########
% delT_M13_2end = max(TR - total_event_duration, 0);
delT_M13_2end = gap_between_readouts - etl - delT_M1_M2;
 
if time2relax_at_the_end > 0
    delT_M13_2end = delT_M13_2end + time2relax_at_the_end;
end

disp(['time to relax at the end of TR: ', num2str(delT_M13_2end), ' sec'])


Mz_mtx = zeross([11, num_voxels, num_reps]);
Mxy_mtx = zeross([9, num_voxels, num_reps]);


Mstart = M0;

tic
for reps = 1:num_reps
    disp(['repetition: ', num2str(reps)])
    
    M1 = relax_T1(M0, Mstart, delT_M0_M1, t1_vals);
        
    % BB: hack
%     M2 = relax_T2(M1, delT_M1_M2, t2_vals);
%     M2 = relax_T2(M1, delT_M1_M2 - 10e-3, t2_vals);
%     M2 = relax_T2(M1, delT_M1_M2 - 9.7e-3, t2_vals);
    M2 = relax_T2_b1(M1, delT_M1_M2 - 9.7e-3, t2_vals, b1, t1_vals);
     
    % acq1
    Mz = M2;
    Mxy = zeros(tf, num_voxels);

    % BB: hack
    % time = 0;
    % add T1 relaxation during T2 prep crusher:
    time = 9.7e-3;

    for q = 1:tf
        Mz = relax_T1(M0, Mz, time, t1_vals);
        
        Mxy(q,:) = relax_T2s(sin(alpha) .* Mz,TE_flash,t2s_vals);
        
        Mz = cos(alpha) .* Mz;

        time = esp;
    end
 
    M3 = Mz;
    Mxy_acq1 = Mxy;
    

    M4 = relax_T1(M0, M3, delT_M3_M4, t1_vals);

%     M5 = -M4;
    % inversion efficiency 
%     M5 = -M4 .* inv_eff;
    % use lookup table
    M5 = inverse_eff(M4,inv_eff);

    M6 = relax_T1(M0, M5, delT_M5_M6, t1_vals);

    
    % acq2
    Mz = M6;
    Mxy = zeros(tf, num_voxels);

    time = 0;

    for q = 1:tf
        Mz = relax_T1(M0, Mz, time, t1_vals);
        
        Mxy(q,:) = relax_T2s(sin(alpha) .* Mz,TE_flash,t2s_vals);
        
        Mz = cos(alpha) .* Mz;

        time = esp;
    end
 
    M7 = Mz;
    Mxy_acq2 = Mxy;
    
    
    
    M8 = relax_T1(M0, M7, delT_M7_M8, t1_vals);

    
    % acq3
    Mz = M8;
    Mxy = zeros(tf, num_voxels);

    time = 0;

    for q = 1:tf
        Mz = relax_T1(M0, Mz, time, t1_vals);


        Mxy(q,:) = relax_T2s(sin(alpha) .* Mz,TE_flash,t2s_vals);
        
        Mz = cos(alpha) .* Mz;

        time = esp;
    end
 
    M9 = Mz;
    Mxy_acq3 = Mxy;
    
    
    M10 = relax_T1(M0, M9, delT_M9_M10, t1_vals);

    
    % % acq4
    % Mz = M10;
    % Mxy = zeros(tf, num_voxels);
    % 
    % time = 0;
    % 
    % for q = 1:tf
    %     Mz = relax_T1(M0, Mz, time, t1_vals);
    % 
    %     Mxy(q,:) = sin(alpha) * Mz;
    % 
    %     Mz = cos(alpha) * Mz;
    % 
    %     time = esp;
    % end
    % 
    % M11 = Mz;
    % Mxy_acq4 = Mxy;
    % 
    % 
    % M12 = relax_T1(M0, M11, delT_M11_M12, t1_vals);

    
    % acq 4
    Mz = M10;
    Mxy = zeros(tf, num_voxels);% times2 because only store 1st echo
    Mxyecho = zeros(tf,num_voxels,6);
    time = 0;

    % for q = 1:tf
    %     Mz = relax_T1(M0, Mz, time, t1_vals);
    % 
    %     Mxy(q,:) = sin(alpha) * Mz;
    % 
    %     Mz = cos(alpha) * Mz;
    % 
    %     time = esp;
    % end
    for q = 1:tf
        Mz = relax_T1(M0, Mz, time, t1_vals);
        
        Mxy(q,:) = sin(alpha) .* Mz;% Mxy13
        
        Mxy0 = sin(alpha) .* Mz;
        Mxyecho(q,:,:) = relax_T2s(Mxy0,TEs,t2s_vals);
        % tmp = relax_T2s(Mxy0,TEs,t2s_vals);
        % Mxyecho(q,:,1)=Mxy0;
        % Mxyecho(q,:,2:end) = tmp(:,2:end);

        Mz = cos(alpha) .* Mz;

        time = TR_mte;
        
        % for e = 1:6
        %     if e==1
        %         Mxy0 = Mxy(q,:);
        %         TE = TEs(1);
        %     else
        %         TE = esp_mte;
        %     end
        %     Mxyecho(q,e,:) = relax_T2s(Mxy0,TE,t2s_vals);
        %     Mxy0 =relax_T2s(Mxy0,TE,t2s_vals);
        % end

    end
 
    M13 = Mz;
    % Mxy_acq5 = Mxy;
    
    
    Mstart = relax_T1(M0, M13, delT_M13_2end, t1_vals);
    
    
    disp(['mean Mz: ', num2str(mean(Mstart))])
    
    Mz_mtx(:,:,reps) = cat(1, M1.', M2.', M3.', M4.', M5.', M6.', M7.', M8.', M9.', M10.', M13.');
    Mxy_mtx(:,:,reps) = cat(1, Mxy_acq1(echo2use,:), Mxy_acq2(echo2use,:), Mxy_acq3(echo2use,:), reshape(Mxyecho(end,:,1),[1,num_voxels]),reshape(Mxyecho(end,:,2),[1,num_voxels]),...
        reshape(Mxyecho(end,:,3),[1,num_voxels]),reshape(Mxyecho(end,:,4),[1,num_voxels]),reshape(Mxyecho(end,:,5),[1,num_voxels]),reshape(Mxyecho(end,:,6),[1,num_voxels]));
end
toc



end

