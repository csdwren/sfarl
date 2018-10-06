function cid = getClass(x)
% It returns the basic type of the input.
if isa(x,'gpuArray')
  cid = classUnderlying(x);
else
  cid = class(x);
end
