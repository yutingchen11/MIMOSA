function [ res ] = svd_apply2d( in, comp_mtx )
% cc: svd coil compression for 2d data
% assumes that coil axis is the 3th dimension
% revised for m3d version where in:(x y z ch 1), for 2d version,in: x y ch 1

mtx_size = size(in(:,:,1));
 
temp = reshape(in, prod(mtx_size), []);

res = reshape(temp * comp_mtx, [mtx_size, size(comp_mtx,2)]);

end