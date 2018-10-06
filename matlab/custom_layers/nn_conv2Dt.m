function [y,dh,db,da] = nn_conv2Dt(x,h,b,alpha,dzdy,varargin)
%   CNN Transpose 2D Convolution 
%   Y = NN_CONV2DT(X,F,B,A) computes the transpose convolution of the 
%   image X with the filter bank A.*F and and biases B. 
%
%   X is a SINGLE array of dimension H x W x D x N where (H,W) are
%   the height and width of the image stack, D is the image depth
%   (number of feature channels) and N the number of of images in the
%   stack.
%
%   F is a SINGLE array of dimension FW x FH x K x FD where (FH,FW)
%   are the filter height and width, K the number of filters in the
%   bank, and FD the depth of a filter (the same as the depth of
%   image X). Filter k is givenby elements F(:,:,k,:); this differ
%   from NN_CONV2D() where a filter is given by elements
%   F(:,:,:,k). FD must be the same as the input depth D.
%
%   B is a SINGLE array with 1 x 1 x K elements (B can in fact
%   be of any shape provided that it has K elements).
%
%   [DX,DF,DB,DA] = NN_CONV2DT(X, F, B, A, DY) computes the derivatives of
%   the operator projected onto P. DX, DF, DB, DA and DY have the same
%   dimensions as X, F, B, A and Y, respectively. %
%
%   NN_CONV2DT(..., 'option', value, ...) accepts the following
%   options:
%
%   `Stride`:: 1
%     The output stride or downsampling factor. If the value is a
%     scalar, then the same stride is applied to both vertical and
%     horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%     allows specifying different downsampling factors for each
%     direction.
%
%   'derParams' :: [true,true,true]
%     If any entry of derParams is set to false then in the backward step 
%     the corresponding parameter df, db or da is not computed.
%
%   `PadSize`:: [0,0,0,0] 
%     Specifies the amount of symmetric padding of the input X as 
%     [TOP, BOTTOM, LEFT, RIGHT].
%
%   `PadType`:: 'zero'
%     Specifies the type of padding. Valid choices are 'zero'|'symmetric'.
%   
%   `cuDNN`:: {'CuDNN'} | 'NoCuDNN'
%     It indicates whether CuDNN will be used or not during the computation
%     of the convolutions.
%      
%   `WeightNormalization` :: false
%      If set to true then the filters are normalized as F/||F||.
%
%   `zeroMeanFilters` :: false
%      If set to true then the mean of the filters is subtracted before
%      applied to the input, i.e, F-E(F), where E is the mean operator.

opts.stride = [1 1];
opts.padSize = [0,0,0,0];
opts.padType = 'zero';
opts.derParams = true(1,3);
opts.cuDNN = 'cuDNN';
opts.zeroMeanFilters = false;
opts.weightNormalization = false;

opts = vl_argparse(opts,varargin);

assert(ismember(opts.padType,{'zero','symmetric'}),['nn_conv2Dt:: ' ...
  'Invalid value for ''padType'' argument.']);

sz_h = size(h);
sz_h = [sz_h ones(1,4-ndims(h))];

if numel(opts.derParams) ~= 3
  opts.derParams=[opts.derParams(:)',false(1,3-numel(opts.derParams))];
end

flag = 0;
tmp = h(:,:,:,1);
if opts.zeroMeanFilters
  h = bsxfun(@minus,h,sum(sum(sum(h,3),2),1)/prod(sz_h(1:3)));
end

if size(h,4) == size(h,1)*size(h,2)
%     tmph = h(:,:,:,1);
%     if sum(tmph(:)) == 0
        h(:,:,:,1) = tmp;
        flag = 1;
%     end
end

if opts.weightNormalization 
  h_norm = sqrt(sum(sum(sum(h.^2,3),2),1));
else
  h_norm = 1;
end

if isempty(alpha)
  alpha = ones(1,1,1,sz_h(4),'like',x);
  opts.derParams(3) = false;
end

num_a = numel(alpha);
assert(ismember(num_a,[1,sz_h(4)]),['nn_conv2Dt:: Invalid dimensions of ' ...
  'the fourth input argument.']);

sz_a = size(alpha);
alpha = reshape(alpha,1,1,1,num_a);

nP = numel(opts.padSize);
if nP == 1
  opts.padSize = opts.padSize(1)*ones(1,4);
end

useSymPad =  (sum(opts.padSize) ~= 0) && isequal(opts.padType,'symmetric');

if nargin < 5 || isempty(dzdy)
  dh = [];
  if opts.weightNormalization
    h = bsxfun(@rdivide,h,h_norm);
  end
  
  pad = [0,0,0,0];
  if isequal(opts.padType,'zero')
    pad = opts.padSize;
  end
  y = vl_nnconvt(x,bsxfun(@times,h,alpha),b,'upsample',opts.stride, ...
    'crop',pad,opts.cuDNN);
  
  if useSymPad
    y = nn_padt(y,opts.padSize);
  end
  
else
  da = [];
  
  if ~opts.derParams(1), opts.derParams(3) = false; end % We need dh in
  % order to compute da.
  
  if opts.weightNormalization
    h = bsxfun(@rdivide,h,h_norm);
  end  
  derOpts={opts.cuDNN};
  if ~opts.derParams(2) || isempty(b)
    derOpts = [derOpts {'NoDerBiases'}];
  end
  if ~opts.derParams(1)
    derOpts = [derOpts {'NoDerFilters'}];
  end
    
  if useSymPad
    dzdy = nn_padt([],opts.padSize,dzdy);
    opts.padSize = [0,0,0,0];
  end
    
  [y,dh,db] = vl_nnconvt(x,bsxfun(@times,h,alpha),b,dzdy,'upsample', ...
    opts.stride,'crop',opts.padSize,derOpts{:}); 
  
  if opts.derParams(3)
    da = sum(sum(sum(h.*dh,3),2),1);
    da = reshape(da,sz_a);
  end

  if opts.derParams(1)
    if opts.weightNormalization
      dh = dh - bsxfun(@times,h,sum(sum(sum(h.*dh,3),2),1));
      tmp = dh(:,:,:,1);
      if opts.zeroMeanFilters
        dh = bsxfun(@minus,dh,sum(sum(sum(dh,3),2),1)/prod(sz_h(1:3)));
      end
      if flag ==1
          dh(:,:,:,1) = tmp;
      end
      dh = bsxfun(@times,dh,alpha./h_norm);
    elseif opts.zeroMeanFilters
      tmp = dh(:,:,:,1);
        dh = bsxfun(@minus,dh,sum(sum(sum(dh,3),2),1)/prod(sz_h(1:3)));
     if flag ==1
          dh(:,:,:,1) = tmp;
      end
        dh = bsxfun(@times,dh,alpha);
    else
      dh = bsxfun(@times,dh,alpha);
    end      
  end
end
  
  
