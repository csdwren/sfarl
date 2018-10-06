function padSize=getPadSize(sizeX,patchSize,stride)
% function getPadSize computes the necessary padding so that an integer
% number of overlapping patches (of size patchSize) can be extracted from 
% the image using a specified stride.
%
%   `Padsize`:: Specifies the amount of padding of an image X as 
%   [TOP, BOTTOM, LEFT, RIGHT]. 

if numel(sizeX) > 2
  sizeX = sizeX(1:2);
end
patchSize = patchSize(1:2);

patchDims = (sizeX-patchSize)./stride + 1;
if any(patchDims <= 0)
  error(['The specified size of the patch is greater than the ' ...
    'dimensions of the image.']);
end  
  
usePad = patchDims - floor(patchDims);

if usePad(1)
  padSizeTB = floor(patchDims(1)).*stride(1) + patchSize(1) - sizeX(1);  
  padSizeTB = [floor(padSizeTB/2) ceil(padSizeTB/2)];
else
  padSizeTB = [0 0];
end
  
if usePad(2)
  padSizeLR = floor(patchDims(2)).*stride(2) + patchSize(2) - sizeX(2);  
  padSizeLR = [floor(padSizeLR/2) ceil(padSizeLR/2)];
else
  padSizeLR = [0 0];
end

padSize=[padSizeTB padSizeLR];