function C=gen_idx_4shrink(inputSize,classtype,isgpu)

if nargin < 2
  isgpu=false;
  classtype='single';
elseif nargin < 3
  isgpu=false;
end

inputSize=inputSize(:)';

if numel(inputSize) < 4
  inputSize=[inputSize ones(1,4-numel(inputSize))];
end

if isgpu
  I=gpuArray.ones(inputSize(1),inputSize(2),classtype);
  colon_=@gpuArray.colon;
else
  I=ones(inputSize(1),inputSize(2),classtype);
  colon_=@colon;
end

C=reshape(repmat(kron(colon_(cast(1,classtype),cast(inputSize(3),classtype)),I),[1 1 inputSize(4)]),inputSize);
