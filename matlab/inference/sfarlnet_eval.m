function res = sfarlnet_eval(net, x, dzdy, res, varargin)
%NET_EVAL  Evaluate the network.
%
%   RES = NET_EVAL(NET, X) evaluates the convnet NET on data X.
%   RES = NET_EVAL(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY.
%   RES = NET_EVAL(NET, X, [], RES) evaluates the NET on X reusing the
%   structure RES.
%   RES = NET_EVAL(NET, X, DZDY, RES) evaluates the NET on X and its
%   derivatives reusing the structure RES.
%
%   The format of the network structure NET and of the result
%   structure RES are described in some detail below. Most networks
%   expect the input data X to be standardized, for example by
%   rescaling the input image(s) and subtracting a mean. Doing so is
%   left to the user, but information on how to do this is usually
%   contained in the `net.meta` field of the NET structure (see
%   below).
%
%   Networks can run either on the CPU or GPU. Use NET_MOVE()
%   to move the network parameters between these devices.
%
%   NET_EVAL(NET, X, DZDY, RES, 'OPT', VAL, ...) takes the following
%   options:
%
%   `Mode`:: `normal`
%      Specifies the mode of operation. It can be either `'normal'` or
%      `'test'`. In test mode, dropout and batch-normalization are
%      bypassed. Note that, when a network is deployed, it may be
%      preferable to *remove* such blocks altogether.
%
%   `ConserveMemory`:: `false`
%      Aggressively delete intermediate results. This in practice has
%      a very small performance hit and allows training much larger
%      models. However, it can be useful to disable it for
%      debugging. Keeps the values in `res(1)` (input) and `res(end)`
%      (output) with the outputs of `loss` and `softmaxloss` layers.
%      It is also possible to preserve individual layer outputs
%      by setting `net.layers{...}.precious` to `true`.
%      For back-propagation, keeps only the derivatives with respect to
%      weights.
%
%   `CuDNN`:: `true`
%      Use CuDNN when available.
%
%   `Accumulate`:: `false`
%      Accumulate gradients in back-propagation instead of rewriting
%      them. This is useful to break the computation in sub-batches.
%      The gradients are accumulated to the provided RES structure
%      (i.e. to call VL_SIMPLENN(NET, X, DZDY, RES, ...).
%
%   `SkipForward`:: `false`
%      Reuse the output values from the provided RES structure and compute
%      only the derivatives (backward pass).
%

%   ## The result format
%
%   NET_EVAL returns the result of its calculations in the RES
%   structure array. RES(1) contains the input to the network, while
%   RES(2), RES(3), ... contain the output of each layer, from first
%   to last. Each entry has the following fields:
%
%   - `res(i+1).x`: the output of layer `i`. Hence `res(1).x` is the
%     network input.
%
%   - `res(i+1).aux_fwd`: any auxiliary output data of layer i during the
%      forward pass.
%
%   - `res(i).aux_bwd`: any auxiliary output data of layer i+1 during the
%      backward pass.
%
%   - `res(i+1).dzdx`: the derivative of the network output relative
%     to the output of layer `i`. In particular `res(1).dzdx` is the
%     derivative of the network output with respect to the network
%     input.
%
%   - `res(i+1).dzdw`: a cell array containing the derivatives of the
%     network output relative to the parameters of layer `i`. It can
%     be a cell array for multiple parameters.
%
%   ## The network format
%
%   The network is represented by the NET structure, which contains
%   two fields:
%
%   - `net.layers` is a cell array with the CNN layers.
%
%   - `net.meta` is a grab-bag of auxiliary application-dependent
%     information, including for example details on how to normalize
%     input data, the class names for a classifiers, or details of
%     the learning algorithm. The content of this field is ignored by
%     NET_EVAL().
%


opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false ;
opts.parameterServer = [] ;
opts.holdOn = false ;

%--Net auxiliary params (Specific for dprnet)--%
opts.netParams = struct('Obs',[], ...
                        'stdn',[], ...
                        'data_mu',[], ...
                        'step',[], ...
                        'origin',[]);
%
% Obs : Indicates the input of the first stage of the network
% stdn : standard deviation for the noise distorting each image 
% in the batch
% data_mu : The data used for creating the look-up table in RBFs
% opts.step : The step size for sampling the domain of the function that is
% approximated by the RBFs
% opts.origin : Defines the lower end from which we start the sampling.
% ---------------
opts = vl_argparse(opts, varargin);

n = numel(net.layers);
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
backPropLim = max(n - opts.backPropDepth + 1, 1);

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
  if opts.skipForward
    error('net_eval:skipForwardNoBackwPass', ...
      '`skipForward` valid only when backward pass is computed.');
  end
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
  bnormCudnn = {'NoCuDNN'} ; % ours seems slighty faster
else
  cudnn = {'NoCuDNN'} ;
  bnormCudnn = {'NoCuDNN'} ;
end

switch lower(opts.mode)
  case 'normal'
    testMode = false ;
  case 'test'
    testMode = true ;
  otherwise
    error('Unknown mode ''%s''.', opts. mode) ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  if opts.skipForward
    error('net_eval:skipForwardEmptyRes', ...
    'RES structure must be provided for `skipForward`.');
  end  
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ... 
    'stats', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1)));
end

if ~opts.skipForward
  res(1).x = x ;
end

if isempty(opts.netParams.Obs) % If Obs is empty then we assume 
  % that the distorted observations match the input of the network.
  opts.netParams.Obs = x;
end

clear x;

% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
  if opts.skipForward, break; end;
  l = net.layers{i};
  res(i).time = tic;
  
  
  switch l.type
      
    case 'clip'
      res(i+1).x = nn_clip(checkInput(res(i).x), l.lb, l.ub);      
          
    case 'sfarlnet'
        
%       [y,~,~,~,~,~,~,~,~,~,~,~,~,J,J_data,M,M_data] = sfarlnet(x,obs,h,h_data,[],[],s,s_data,[],[],rbf_weights,rbf_weights_data,rbf_means,...
%   rbf_precision,stdn,alpha,lambda,[],'stride',stride,'padSize',padSize,'padSize_data',padSize_data,...
%   'padType',padType,'zeroMeanFilters',zeroMeanFilters,...
%   'weightNormalization',weightNormalization,'data_mu',data_mu,'conserveMemory',false);

% 'weights', {{opts.h, opts.h_data, opts.ht, ...
%       opts.ht_data, opts.s, opts.s_data, opts.st, opts.st_data...
%       opts.rbf_weights,opts.rbf_weights_data,opts.alpha,opts.lambda}},...

      if doder
        [res(i+1).x,~,~,~,~,~,~,~,~,~,~,~,~,res(i+1).aux{1},res(i+1).aux{2},res(i+1).aux{3},res(i+1).aux{4}] = ...
          sfarlnet(checkInput(res(i).x), ...
          opts.netParams.Obs, l.weights{1}, l.weights{2}, l.weights{3}, ...
          l.weights{4}, l.weights{5}, l.weights{6}, l.weights{7}, l.weights{8},...
          l.weights{9},l.weights{10}, l.rbf_means, l.rbf_precision, ...
          opts.netParams.stdn, l.weights{11}, l.weights{12}, [], 'conserveMemory', ...
          false, 'stride', l.stride, 'padSize', l.padSize, 'padSize_data',l.padSize_data,...
          'padType', l.padType,'zeroMeanFilters',l.zeroMeanFilters, ...
          'weightNormalization',l.weightNormalization, 'cuDNN',cudnn{:}, ...
          'data_mu',opts.netParams.data_mu,'step',opts.netParams.step, ...
          'origin',opts.netParams.origin, 'shrink_type',l.shrink_type, ...
          'lb', l.lb, 'ub', l.ub);
      else
        res(i+1).x = sfarlnet(checkInput(res(i).x), opts.netParams.Obs, ...
          l.weights{1}, l.weights{2}, l.weights{3}, ...
          l.weights{4}, l.weights{5}, l.weights{6}, l.weights{7}, l.weights{8},...
          l.weights{9}, l.weights{10}, l.rbf_means, l.rbf_precision, ...
          opts.netParams.stdn, l.weights{11}, l.weights{12}, [], 'conserveMemory', ...
          true, 'stride', l.stride, 'padSize', l.padSize,'padSize_data',l.padSize_data, 'padType', ...
          l.padType,'zeroMeanFilters',l.zeroMeanFilters, ...
          'weightNormalization',l.weightNormalization, 'cuDNN',cudnn{:}, ...
          'data_mu',opts.netParams.data_mu,'step',opts.netParams.step, ...
          'origin',opts.netParams.origin, 'shrink_type',l.shrink_type, ...
          'lb', l.lb, 'ub', l.ub);
      end          
      
    otherwise
      error('Unknown layer type ''%s''.', l.type) ;
  end
  
  % optionally forget intermediate results
  needsBProp = doder && i >= backPropLim;
  forget = opts.conserveMemory && ~needsBProp;  
  if i > 1
    lp = net.layers{i-1};
    forget = forget & ~strcmp(lp.type, 'imloss');
    if isfield(lp,'precious') 
      forget = forget & ~lp.precious; 
    end
  end

  if forget 
    res(i).x = [] ;
  end
  
  if gpuMode && opts.sync
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

if opts.conserveMemory && backPropLim > n % If there is not a backward pass
  % we delete all the intermediate results at the end of the forward pass.
  for i = 1:n
    res(i).x = [];
  end
end


% -------------------------------------------------------------------------
%    Backward pass
%   will come soon ...
% -------------------------------------------------------------------------


function x = checkInput(x)
% It checks if the input is a cell-array. If this is the case it returns
% the last element of the cell. Otherwise it does nothing.
if iscell(x)
  x = x{end};
end

% % In case the last entry of the cell is a cell itself
% while iscell(x)
%   x = x{end};
% end

% If the second input is isempty then the output equals to the first input.
% Otherwise the output equals to the second input.