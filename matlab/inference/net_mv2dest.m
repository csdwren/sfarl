function net = net_mv2dest(net, destination)
%NET_MV2DEST  Move a CNN network between CPU and GPU.
%   NET = NET_MV2DEST(NET, 'gpu') moves the network to the
%   current GPU device. NET = NET_MOVE(NET, 'cpu') moves the
%   network to the CPU.


switch destination
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  case 'double', moveop = @(x) double(x);
  case 'single', moveop = @(x) single(x);
  otherwise, error('Unknown destination ''%s''.', destination) ;
end

for l=1:numel(net.layers)

  if isfield(net.layers{l}, 'filters')
    for j=1:numel(net.layers{l}.filters)
      net.layers{l}.filters{j} = moveop(net.layers{l}.filters{j}) ;
    end
  end
  
  if isfield(net.layers{l}, 'weights')
    for j=1:numel(net.layers{l}.weights)
      net.layers{l}.weights{j} = moveop(net.layers{l}.weights{j}) ;
    end
  end

  if isfield(net.layers{l},'rbf_means')
    net.layers{l}.rbf_means=moveop(net.layers{l}.rbf_means);
  end
  
  if isfield(net.layers{l}, 'Idx')
    net.layers{l}.Idx=moveop(net.layers{l}.Idx);
  end
  
end
