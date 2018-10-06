function [y,mask] = nn_clip(x,lb,ub,dzdy,varargin)
%NN_CLIP CNN clipping layer.
%   Y = NN_CLIP(X) clips the values of X that are outsize the [lb,ub] 
%   interval and sets them equal to the boundaries of the interval, so that 
%   it holds lb <= clip(X) <= ub. X can have arbitrary size. 
%
%   DZDX = NN_CLIP(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%   NN_CLIP(...,'OPT',VALUE,...) takes the following options:
%
%   `LB`:: 0
%      Sets the lower boundary constraint.
%   `UB`:: 255
%      Sets the upper boundary constraint.
%    `MASK`:: A mask which indicates in which locations the input does not
%    satify the lower and upper bounds. It is estimated in the forward pass
%    and used in the backward pass for the computation of the gradient.

% stamatis@math.ucla.edu 26/01/2016.

opts.mask = [];
opts = vl_argparse(opts,varargin);

if nargin < 2, lb = 0; end
if nargin < 3, ub = 255; end

% if lb==-inf and ub==inf then no clipping takes place.
do_clip = ~(isequal(lb,-inf) &&  isequal(ub,inf));

mask = [];
if nargin < 4 || isempty(dzdy)
  if do_clip
    y = max(min(ub,x),lb);
    if nargout > 1
      mask = find(x > ub);
      mask = [mask; find(x < lb)];
    end
  else
    y = x;
  end
else
  y = dzdy;
  
  if do_clip
    if ~isempty(x)
      y(x > ub) = 0;
      y(x < lb) = 0;
    else
      y(opts.mask) = 0;
    end
  end
end