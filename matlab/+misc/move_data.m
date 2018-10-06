function varargout = move_data(destination, varargin)

err_msg = 'The number of outputs cannot exceed the number of inputs';
assert(nargout <= nargin, err_msg);

varargout = cell(nargout,1);

switch destination
  case 'cpu'    
    for k=1:nargout
      if isstruct(varargin{k})
        varargout{k} = varargin{k};
        fn = fieldnames(varargout{k});
        for i=1:numel(fn)
          if isa(varargout{k}.(char(fn{i})),'function_handle'), continue; end
          varargout{k}.(char(fn{i})) = gather(varargout{k}.(char(fn{i})));
        end
      else
        varargout{k}=gather(varargin{k});
      end
    end
  case 'gpu'
    for k=1:nargout
      if isstruct(varargin{k})
        varargout{k} = varargin{k};
        fn = fieldnames(varargout{k});
        for i=1:numel(fn)
          if isa(varargout{k}.(char(fn{i})),'function_handle'), continue; end
          varargout{k}.(char(fn{i})) = gpuArray(varargout{k}.(char(fn{i})));
        end
      else
        varargout{k}=gpuArray(varargin{k});
      end
    end
  otherwise
    error('Unknown destination');
end