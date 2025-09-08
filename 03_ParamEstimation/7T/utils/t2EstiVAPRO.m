function [t2Image, mImage] = t2EstiVAPRO(imageStack, teValues, possibleT2Values)

% Joint estimation of spin density and T2 from a sequence of T2 weighted
% images with the variable projection (VARPRO) method.
% Inputs:
% imageStack: T2 weighted image sequence: Echo dim * Voxel dim
% teValues: Echo times
% possibleT2Values: the range of possible T2 values

% Bo Zhao (zhaobouiuc@gmail.com)
% April 3, 2015

    disp('T2 estimation by VAPRO');
    t2Image = zeros(size(imageStack, 2), 1);
    if (nargout >1)
        mImage = zeros(size(imageStack, 2), 1);
    end
    Nt2 = numel(possibleT2Values);
    Nvox = size(imageStack, 2);

    possibleR2Values = 1./possibleT2Values;
    Tcolumn2 = exp(-possibleR2Values(:)*(reshape(teValues, 1, [])));
    costVal = zeros(Nt2, Nvox);
    Tcol_length = zeros(Nt2, 1);
    for index1 = 1:Nt2
        Tcol_length(index1) = norm(Tcolumn2(index1, :));
    end

    for index1 = 1:Nvox  
        costVal(:, index1) = abs(Tcolumn2*imageStack(:, index1))./Tcol_length;
        [~, indices]       = max(costVal(:, index1), [], 1);
        t2Image(index1)    = possibleT2Values(indices);
        [q, r]             = qr(reshape(Tcolumn2(indices, :), [], 1), 0);
        mImage(index1)     = q'*imageStack(:, index1)/r;
    end
