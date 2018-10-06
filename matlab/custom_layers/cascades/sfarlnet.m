function [y,dh1,dh1_data,dh2,dh2_data,ds1,ds1_data,ds2,ds2_data,dw,dw_data,da,dl,J,J_data,M,M_data] ...
    = sfarlnet(x,Obs,h1,h1_data,h2,h2_data,s1,s1_data,s2,s2_data,...
  rbf_weights,rbf_weights_data,rbf_means,rbf_precision,stdn,alpha,lambda,dzdy,varargin)
%SFARL
%
%  If h2 is empty then the 1st and 4th layers share the same parameters h1
%
%  x    : input of the current stage
%  Obs  : input of the first stage of the network.
%  stdn : not used
%  alpha: not used
%  h1,h2: Filters for nn_conv2D and nn_conv2Dt in regularization term
%  s1,s2: Scaling coefficients for nn_conv2D and nn_conv2Dt in regularization term
%  h1_data,h2_data: Filters for nn_conv2D and nn_conv2Dt in fidelity term
%  s1_data,s2_data: Scaling coefficients for nn_conv2D and nn_conv2Dt in fidelity term
%


opts.stride = [1,1];
opts.padSize = [0,0,0,0];
opts.padSize_data = [0,0,0,0];
opts.padType = 'symmetric';
opts.cuDNN = 'cuDNN';
opts.Jacobian = [];
opts.Jacobian_data = [];
opts.clipMask = [];
opts.clipMask_data = [];
opts.Idx=[];
opts.conserveMemory = false;
opts.first_stage = 0; % Denotes if this the first stage of the network.
opts.learningRate = [1,1,1,1,1,1];
opts.zeroMeanFilters = false;
opts.weightNormalization = false;
opts.data_mu = [];
opts.step = 0.2;
opts.origin = [];
opts.shrink_type = 'identity';
opts.lb = -100;
opts.ub = 100;
%-----------------------------
opts = vl_argparse(opts,varargin);

if numel(opts.learningRate) ~= 6
  opts.learningRate = [opts.learningRate(:)' ones(1,12-numel(opts.learningRate))];
end

switch opts.shrink_type
  case 'identity'
    Shrink = @nn_grbfShrink_lut;
  case 'residual'
    Shrink = @nn_grbfResShrink_lut;
  otherwise
    error('sfarlet :: Unknown type of RBF shrinkage.');
end

assert(size(h1,4) == size(rbf_weights,1), ['Invalid input for ' ...
  'h1 - dimensions mismatch.']);
assert(isempty(h2) || all(size(h1) == size(h2)) , ['Invalid input for ' ...
  'h2 - dimensions mismatch.']);

assert(size(h1_data,4) == size(rbf_weights_data,1), ['Invalid input for ' ...
  'h1 in fidelity term - dimensions mismatch.']);
assert(isempty(h2_data) || all(size(h1_data) == size(h2_data)) , ['Invalid input for ' ...
  'h2 in fidelity term - dimensions mismatch.']);

weightSharing = isempty(h2);% If h2 is empty then conv2D and conv2Dt
% share the same weights.

if nargin < 18 || isempty(dzdy)
  dh1=[];dh2=[];ds1=[];ds2=[];dw=[];da=[];J=[];M=[];
  dh1_data=[];dh2_data=[];ds1_data=[];ds2_data=[];dw_data=[];
  da=[];dl=[];
  

  if opts.conserveMemory
    
      %% regularization term
    y = nn_conv2D(x,h1,[],s1,[],'stride',opts.stride,'padSize',...
      opts.padSize,'padType',opts.padType,'cuDNN',opts.cuDNN,...
      'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
      opts.weightNormalization);
    
    y = nn_clip(y,opts.lb,opts.ub);    
    y = Shrink(y,rbf_weights,rbf_means,rbf_precision,[],...
      'Idx',opts.Idx,'data_mu',opts.data_mu,'step',opts.step,'origin',...
      opts.origin);
    
    if weightSharing
      y = nn_conv2Dt(y,h1,[],s1,[],'stride',opts.stride,'padSize',...
        opts.padSize,'padType',opts.padType,'cuDNN', opts.cuDNN,...
        'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
        opts.weightNormalization);
    else
      y = nn_conv2Dt(y,h2,[],s2,[],'stride',opts.stride,'padSize',...
        opts.padSize,'padType',opts.padType,'cuDNN', opts.cuDNN,...
        'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
        opts.weightNormalization);
    end
    
    %% fidelity term
    res_input = x - Obs; % residual image
    y_data = nn_conv2D(res_input,h1_data,[],s1_data,[],'stride',opts.stride,'padSize',...
      opts.padSize_data,'padType',opts.padType,'cuDNN',opts.cuDNN,...
      'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
      opts.weightNormalization);
    
    y_data = nn_clip(y_data,opts.lb,opts.ub);    

%     if nargout > 14
%       [y_data,~,J_data] = Shrink(y_data,rbf_weights_data,rbf_means,rbf_precision,[],...
%         'Idx',opts.Idx,'data_mu',opts.data_mu,'step',opts.step,'origin',...
%         opts.origin);
%     else
      y_data = Shrink(y_data,rbf_weights_data,rbf_means,rbf_precision,[],...
        'Idx',opts.Idx,'data_mu',opts.data_mu,'step',opts.step,'origin',...
        opts.origin);
%     end
    
    if weightSharing
      y_data = nn_conv2Dt(y_data,h1_data,[],s1_data,[],'stride',opts.stride,'padSize',...
        opts.padSize_data,'padType',opts.padType,'cuDNN', opts.cuDNN,...
        'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
        opts.weightNormalization);
    else
      y_data = nn_conv2Dt(y_data,h2_data,[],s2_data,[],'stride',opts.stride,'padSize',...
        opts.padSize_data,'padType',opts.padType,'cuDNN', opts.cuDNN,...
        'zeroMeanFilters',opts.zeroMeanFilters,'weightNormalization',...
        opts.weightNormalization);
    end
    
    %%
    %y = x - y - lambda*y_data;
    %lambda = lambda*0;
    y_data = vl_nnconv(y_data,lambda,[],'stride',opts.stride, ...
    'pad',0,'dilate',1,opts.cuDNN);
    y = x - y - y_data;
    %%
  
  else
    
    % -------------------------------------------------------------------------
    %%% Forward pass with recording intermediate results
    %%% will come soon
    % -------------------------------------------------------------------------
    
  end
else
  
    % -------------------------------------------------------------------------
    %%% Backward pass
    %%% will come soon
    % -------------------------------------------------------------------------
  
end

