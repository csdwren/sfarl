function net = net_move(net, destination)
%NET_MOVE  Move a CNN network between CPU and GPU.
%   NET = NET_MOVE(NET, 'gpu') moves the network to the
%   current GPU device. NET = NET_MOVE(NET, 'cpu') moves the
%   network to the CPU.


switch destination
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown destination ''%s''.', destination) ;
end

for l=1:numel(net.layers)
    
  if isfield(net.layers{l}, 'weights')
    for j=1:numel(net.layers{l}.weights)
      net.layers{l}.weights{j} = moveop(net.layers{l}.weights{j}) ;
    end
  end
    
end
