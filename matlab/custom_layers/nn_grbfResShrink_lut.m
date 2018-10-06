function [y,dzdw,J] = nn_grbfResShrink_lut(x,weights,means,sigma,dzdy,varargin)

%NN_GRBFResShrink_LUT shrinkage using an RBF-mixture with truncated Gaussian basis
%functions. (To compute the GRBF we use a look-up table and perform linear
%interpolation for the values that do not coincide with the saved ones).
%
%   Y = NN_GRBFResShrink_lut(X,WEIGHTS,MEANS,SIGMA) applies the function
%   approximated by a mixture of Radial Basis fuctions to every element of
%   X.
%
%   If X is of size H x W x K x N (K: number of channels, N: number of
%   images) then weights is of size K x M where M is the number
%   of mixture components in the RBF-mixture, means is of size M x 1 and
%   precision is a scalar.
%
%                         M
%  r(z_(n,k)) = z_(n,k) + S  w_(k,j)*exp(-0.5*(|z_(n,k)-mu_j|/sigma)^2)
%                        j=1
%
%   where n=1:H*W refers to the spatial coordinates of X, k=1:K refers
%   to the channels of X. (w=weights, mu=means).
%
%   [DZDX, DZDW] = NN_GRBFResShrink_LUT(X, WEIGHTS,MEANS,SIGMA, DZDY) computes
%   the derivatives of the block projected onto DZDY. DZDX, DZDW and DZDY
%   have the same dimensions as X, WEIGHTS, and Y respectively.
%
%   NN_GRBFResShrink_LUT(...,'OPT',VALUE,...) takes the following options:
%
%   `Jacobian`:: J which is matrix of the same size as DZDY. J is estimated
%   during the forward pass of the function. If it is not provided then it
%   is re-estimated in the backward pass.
%
%   'delta' :: The distance between two centers of Gaussian basis
%   functions.(Default: sigma)
%
%   'Idx' :: Idx is an array of the same size as x and can be computed
%   using the function Idx=misc.gen_idx_4shrink(size(x));
%
%   'derParams' :: if is set to false then in the backward step dzdw is not
%   computed. (Default value : true)
%

% s.lefkimmiatis@skoltech.ru, 23/05/2016

means = cast(means(:),'like',x);

opts.data_mu = [];
opts.origin = [];%-320;
opts.step = [];%0.2;
opts.Idx = [];
opts.Jacobian = [];
opts.derParams = true;
opts = vl_argparse(opts, varargin) ;

sz=size(x);

if numel(sz) < 4
  sz=[sz ones(1,4-numel(sz))];
end

szW=size(weights);

err_msg=['nn_grbfResShrink_lut:: The first dimension of weights must match ',...
  'the 3rd dimension of x.'];
assert(sz(3)==szW(1),err_msg);

% ------------------ Prepare the look-up table ----------------------------
%
%           M
% L(k,n) = Sum w(k,m)*exp(-0.5*((x(n)-means(m))/sigma)^2), k=1:K, n=1:N
%          m=1
% where N is the number of points that we evaluate the Gaussian RBF.


if isempty(opts.data_mu)  
  if isa(x,'gpuArray')
    data = gpuArray.colon(cast(opts.origin,'like',x),cast(opts.step,'like',x),cast(means(end)+sigma,'like',x));
    means = gpuArray(means);
  else
    data = colon(cast(opts.origin,'like',x),cast(opts.step,'like',x),cast(means(end)+sigma,'like',x));
    means = gather(means);
  end
  opts.data_mu = bsxfun(@minus,data,means);
end

if isa(x,'gpuArray')
  opts.data_mu = gpuArray(opts.data_mu);
else
  opts.data_mu = gather(opts.data_mu);
end

nbins = size(opts.data_mu,2);
%--------------------------------------------------------------------------

if isempty(opts.origin)
  opts.origin = gather(opts.data_mu(1,1)+means(1));
end

misc.checkEq(opts.origin,opts.data_mu(1,1)+means(1),...
  'The provided value for origin is not correct.');
%assert(gather(opts.origin == opts.data_mu(1,1)+means(1)),'The provided value for origin is not correct.');

if isempty(opts.step)
  opts.step = opts.data_mu(1,2)-opts.data_mu(1,1);
end

if nargin <= 4 || isempty(dzdy)
    
  dzdw=[];
  
  % --------------- Prepare the look-up table ----------------------------%  
  L = zeros(szW(1),nbins,'like',x);
  if nargout > 2
    LD = L;
    for k=1:szW(1)
      tmp = bsxfun(@times,exp(-0.5*(opts.data_mu/sigma).^2),weights(k,:)');
      L(k,:)=sum(tmp,1);
      LD(k,:)=sum(-tmp.*opts.data_mu/sigma^2,1);
    end
    clear tmp
  else
    for k=1:szW(1)
      L(k,:)=sum(bsxfun(@times,exp(-0.5*(opts.data_mu/sigma).^2),weights(k,:)'),1);
    end
  end  
  % ----------------------------------------------------------------------%  
  
  if isempty(opts.Idx)
    if isequal(class(x),'gpuArray')
      opts.Idx = misc.gen_idx_4shrink(sz,classUnderlying(x),true);
    else
      opts.Idx = misc.gen_idx_4shrink(sz,class(x),false);
    end
  end
  
  if (size(opts.Idx,4) ~= sz(4))
    opts.Idx = opts.Idx(:,:,:,1:sz(4));
  end

  x_bin = (x - opts.origin)/opts.step;
  % The valid values of the bins are in the range [0, nbins-1].
  idx_l = min(max(floor(x_bin),0),nbins-1); % The index of the left bin 
  % that encloses x
  idx_h = min(max(ceil(x_bin),0),nbins-1); % The index of the right bin  
  % that encloses x
  
  wh = x_bin - idx_l;
  
  % Linear interpolation of f(x) for x that lies in [x1,x2].
  % f(x) = f(x1) + (f(x2)-f(x1))*(x-x1)/(x2-x1) where x2-x1 = step.
  %
  % opts.Idx+idx_l*szW(1) indicates the index of the left bin for the
  % all the x's of the different K channels. The same holds for the right
  % bin.
  y = x + L(opts.Idx+idx_l*szW(1)).*(1-wh) + L(opts.Idx+idx_h*szW(1)).*wh;
   
  if nargout > 2
    J = 1 + LD(opts.Idx+idx_l*szW(1)).*(1-wh) + LD(opts.Idx+idx_h*szW(1)).*wh;
  else
    J = [];
  end  
  
else
  
  if isempty(opts.Jacobian)
    
    % --------------- Prepare the look-up table --------------------------%
    LD = zeros(szW(1),nbins,'like',x);
    for k=1:szW(1)
      tmp = bsxfun(@times,exp(-0.5*(opts.data_mu/sigma).^2),weights(k,:)');
      LD(k,:)=sum(-tmp.*opts.data_mu/sigma^2,1);
    end
    clear tmp
    % --------------------------------------------------------------------%
    
    if isempty(opts.Idx)
      if isequal(class(x),'gpuArray')
        opts.Idx = misc.gen_idx_4shrink(sz,classUnderlying(x),true);
      else
        opts.Idx = misc.gen_idx_4shrink(sz,class(x),false);
      end
    end
    
    if (size(opts.Idx,4) ~= sz(4))
      opts.Idx = opts.Idx(:,:,:,1:sz(4));
    end
    
    x_bin = (x - opts.origin)/opts.step;
    % The valid values of the bins are in the range [0, nbins-1].
    idx_l = min(max(floor(x_bin),0),nbins-1); % The index of the left bin
    % that encloses x
    idx_h = min(max(ceil(x_bin),0),nbins-1); % The index of the right bin
    % that encloses x
    
    wh = x_bin - idx_l;

    y = LD(opts.Idx+idx_l*szW(1)).*(1-wh) + LD(opts.Idx+idx_h*szW(1)).*wh;
    y = (1+y).*dzdy;
  else
    J = [];
    y = opts.Jacobian.*dzdy;
  end
  
  if opts.derParams
    
%     if ~isempty(opts.Jacobian)
%       x_bin = (x - opts.origin)/opts.step;
%       % The valid values of the bins are in the range [0, nbins-1].
%       idx_l = min(max(floor(x_bin),0),nbins-1); % The index of the left bin
%       % that encloses x
%       idx_h = min(max(ceil(x_bin),0),nbins-1); % The index of the right bin
%       % that encloses x
%       wh = x_bin - idx_l;
%     end
%    
%     dzdw = zeros(szW,'like',weights);
%     for m = 1:szW(2)
%       % --------------- Prepare the look-up table ------------------------%
%       LD = exp(-0.5*(opts.data_mu(m,:)/sigma).^2);
%       % ------------------------------------------------------------------%
%       dzdw(:,m) = sum(sum(sum((LD(idx_l+1).*(1-wh)+LD(idx_h+1).*wh).*dzdy,4),2),1);
%     end
    
    dzdw = zeros(szW,'like',weights);
    for m = 1:szW(2)
      tmp = exp(-0.5*((x-means(m))/sigma).^2);
      dzdw(:,m) = sum(sum(sum(tmp.*dzdy,4),2),1);
    end    
    
%     LD = exp(-0.5*(opts.data_mu'/sigma).^2); % N x M where N the number of 
%     % estimated values and M the number of means.
%     I = reshape((0:szW(2)-1)*nbins,1,1,1,szW(2));
%     for n = 1:sz(4)      
%       dzdw = dzdw + reshape(sum(reshape(bsxfun(@times,LD(bsxfun(@plus,idx_l(:,:,:,n)+1,I)),(1-wh(:,:,:,n)))+ ...
%       bsxfun(@times,LD(bsxfun(@plus,idx_h(:,:,:,n)+1,I)),wh(:,:,:,n)),[],szW(1),szW(2)),1),szW);
%     end      
  else
    dzdw = [];
  end
  
end

