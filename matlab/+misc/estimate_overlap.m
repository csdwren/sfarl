function T=estimate_overlap(dimsX,patchSize,stride,varargin)
% This function returns an array of the same size of the spatial dimensions
% of an  image x (dimsX), which indicates to how many patches extracted 
% from the image (patches are of size patchSize and are extracted using a 
% specified stride) each pixel of the image contributes. 

% For example below is the array which indicates how many times each pixel
% at the particular location of an image of size 16 x 16 has been found in 
% any of the 49 4x4 patches that have been extracted using a stride=2.
%
% T = 
%   1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
%   1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   2     2     4     4     4     4     4     4     4     4     4     4     4     4     2     2
%   1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
%   1     1     2     2     2     2     2     2     2     2     2     2     2     2     1     1
%
%
% Based on this table the pixel at the location (3,2) has been used in 4
% different patches while the pixel at the location (15,4) has been used in
% 2 different patches.

opts.useGPU=false;
opts.cuDNN = 'cuDNN';
opts.classType='single';
opts.padSize = [0,0,0,0];
opts=vl_argparse(opts,varargin);

if opts.useGPU
  x=gpuArray.ones(dimsX,opts.classType);
else
  x=ones(dimsX,opts.classType);
end

Pn = prod(patchSize);
h = eye(Pn,'like',x);
h = reshape(h,[patchSize 1 Pn]);

T=vl_nnconv(x,h,[],'stride',stride,'pad',opts.padSize,opts.cuDNN);
T=vl_nnconvt(T,h,[],'upsample',stride,'crop',opts.padSize,opts.cuDNN);

  


