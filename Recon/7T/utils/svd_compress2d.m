function [ res, comp_mtx ] = svd_compress2d( in, num_svd, flip_on )
% cc: svd coil compression for 2d data(revised for m3d version where in:x y z ch 1)
% for 2d version: in is x y ch 1
% assumes that coil axis is the 3th dimension

if nargin < 3
    flip_on = 0;
end

mtx_size = size(in(:,:,1));
 
temp = reshape(in, prod(mtx_size), []);

[v,d] = eig(temp'*temp);

if flip_on
    v = flipdim(v,2);
    comp_mtx = v(:,1:num_svd);    
else
    comp_mtx = v(:,end-num_svd+1:end);
end

res = reshape(temp * comp_mtx, [mtx_size, num_svd]);

end