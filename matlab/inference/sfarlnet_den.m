function out = sfarlnet_den(y,varargin)

opts.stdn = 0;
opts.netPath = fullfile(fileparts(mfilename('fullpath')),'models');
opts.fsize = 7; % The support of the filters
opts = vl_argparse(opts,varargin);


opts = vl_argparse(opts,varargin);

l=load(opts.netPath);
net = l.net;
clear l;

% if net.layers{end}.type == 'imloss'
%     % net.layers{end}=[];
%     net.layers = net.layers(1:end-1);
% end

Params = net.meta.netParams;
Params.stdn = opts.stdn;

if isa(y,'gpuArray')
    cid = classUnderlying(y);
else
    cid = class(y);
end

if strcmp(cid,'double')
    for k=1:numel(net.layers)
        if isfield(net.layers{k},'weights')
            for j=1:numel(net.layers{k}.weights)
                net.layers{k}.weights{j} = double(net.layers{k}.weights{j});
            end
        end
        if isfield(net.layers{k},'rbf_means')
            net.layers{k}.rbf_means = double(net.layers{k}.rbf_means);
        end
    end
    
    Params.data_mu = double(Params.data_mu);
end

if isa(y,'gpuArray')
    net = net_mv2dest(net,'gpu');
    Params = misc.move_data('gpu',Params);
end

out = sfarlnet_eval(net,y,[],[],'netParams',Params,'conserveMemory',true);
out = out(end).x;