function [y,dh,db,da] = nn_conv2D(x,h,b,alpha,dzdy,varargin)
%   CNN 2D Convolution 
%   Y = NN_CONV2D(X,F,B,A) computes the convolution of the image X
%   with the filter bank A.*F and and biases B. 
%
%   X is an array of dimension H x W x C x N where (H,W) are the
%   height and width of the image stack, C is the number of feature
%   channels, and N is the number of images in the batch.
%
%   F is an array of dimension FW x FH x FC x K where (FH,FW) are the
%   filter height and width and K the number o filters in the bank. FC
%   is the number of feature channels in each filter and must match
%   the number of feature channels C in X. Alternatively, FC can
%   *divide* the C; in this case, filters are assumed to form G=C/FC
%   *groups* of equal size (where G must divide K). Each group of
%   filters works on a consecutive subset of feature channels of the
%   input array X.
%
%   [DX,DF,DB,DA] = NN_CONV2D(X, F, B, A, DY) computes the derivatives of
%   the operator projected onto P. DX, DF, DB, DA and DY have the same
%   dimensions as X, F, B, A and Y, respectively. %
%
%   NN_CONV2D(..., 'option', value, ...) accepts the following
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
%   `Dilate`:: 1
%     Set the kernel dilation factor. Passing [DILATEY DILATEX] allows
%     specifying different dilation factors for Y and X. Filters are
%     dilated by inserting DILATE-1 zeros between filter elements. For
%     example, the filter
%
%       [1 3]
%       [2 4]
%
%     is implicitly treated as
%
%       [1 0 3]
%       [0 0 0]
%       [2 0 4]
%
%     by setting DILATE equal to 2.
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
opts.dilate = 1;
opts.derParams = true(1,3);
opts.cuDNN = 'cuDNN';
opts.zeroMeanFilters = false;
opts.weightNormalization = false;

opts = vl_argparse(opts,varargin);

assert(ismember(opts.padType,{'zero','symmetric'}),['nn_conv2D:: ' ...
  'Invalid value for ''padType'' argument.']);

sz_h = size(h);
sz_h = [sz_h ones(1,4-ndims(h))];

if numel(opts.derParams) ~= 3
  opts.derParams=[opts.derParams(:)',false(1,3-numel(opts.derParams))];
end

flag=0;
tmp = h(:,:,:,1);
if opts.zeroMeanFilters
  h = bsxfun(@minus,h,sum(sum(sum(h,3),2),1)/prod(sz_h(1:3)));
end

if size(h,4) == size(h,1)*size(h,2)
%     tmph = h(:,:,:,1);
%     if sum(tmph(:)) == 0
        h(:,:,:,1) = tmp;
        flag=1;
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
assert(ismember(num_a,[1,sz_h(4)]),['nn_conv2D:: Invalid dimensions of ' ...
  'the fourth input argument.']);

sz_a = size(alpha);
alpha = reshape(alpha,1,1,1,num_a);

nP = numel(opts.padSize);
if nP == 1
  opts.padSize = opts.padSize(1)*ones(1,4);
end

useSymPad =  (sum(opts.padSize) ~= 0) && isequal(opts.padType,'symmetric');

if useSymPad
  x = nn_pad(x,opts.padSize);
end


if nargin < 5 || isempty(dzdy)
  dh = [];
  if opts.weightNormalization
    h = bsxfun(@rdivide,h,h_norm);
  end
  
  if useSymPad 
    opts.padSize = [0,0,0,0];
  end
  y = vl_nnconv(x,bsxfun(@times,h,alpha),b,'stride',opts.stride, ...
    'pad',opts.padSize,'dilate',opts.dilate,opts.cuDNN);
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
  
  pad = 0;
  if isequal(opts.padType,'zero')
    pad = opts.padSize;
  end
    
  [y,dh,db] = vl_nnconv(x,bsxfun(@times,h,alpha),b,dzdy,'stride',opts.stride, ...
    'pad',pad,'dilate',opts.dilate,derOpts{:});
  
  if useSymPad
    y = nn_pad([],opts.padSize,y);
  end
  
  if opts.derParams(3)
    da = sum(sum(sum(h.*dh,3),2),1);
    da = reshape(da,sz_a);
  end

  if opts.derParams(1)
    if opts.weightNormalization
      dh = dh - bsxfun(@times,h,sum(sum(sum(h.*dh,3),2),1));
      if opts.zeroMeanFilters
        tmp = dh(:,:,:,1);
          dh = bsxfun(@minus,dh,sum(sum(sum(dh,3),2),1)/prod(sz_h(1:3)));
      if flag==1
          dh(:,:,:,1)=tmp;
      end
      end
      dh = bsxfun(@times,dh,alpha./h_norm);
    elseif opts.zeroMeanFilters
      tmp = dh(:,:,:,1);
        dh = bsxfun(@minus,dh,sum(sum(sum(dh,3),2),1)/prod(sz_h(1:3)));
      if flag==1
          dh(:,:,:,1)=tmp;
      end
        dh = bsxfun(@times,dh,alpha);
    else
      dh = bsxfun(@times,dh,alpha);
    end      
  end
end
  
  
