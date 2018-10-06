function As=shift_inf(A,e,sgn)

%SHIFT_INF Shifting operator with INF boundary conditions
%   B = SHIFT(A,SHIFTSIZE,SGN) shifts the values in the array A by SHIFTSIZE
%   elements. SHIFTSIZE is a vector of integer scalars where the N-th
%   element specifies the shift amount for the N-th dimension of array A.
%   If an element in SHIFTSIZE is positive, the values of A are shifted
%   down (or to the right). If it is negative, the values of A are shifted
%   up (or to the left).
%   IF SGN is given then boundary conditions are equal to SIGN(SGN)*INF

if nargin < 2
  error('shift:NoInputs',['No input arguments specified. ' ...
    'There should be at least two input arguments.'])
end

if nargin < 3
  sgn = 1;
else
  sgn = sign(sgn);
end

[e, sizeA, numDims, msg] = ParseInputs(A,e);

if (~isempty(msg))
  error('shift:InvalidShiftType','%s',msg);
end

if any(sizeA-abs(e) < 0)
  msg=[ 'The step of the shift at each dimension must be '...
    'less or equal than the size of the dimension itself.'];
  error('shift:InvalidShiftType','%s',msg);
end

idx=cell(numDims,1);

As=ones(sizeA,'like',A)*sgn*inf;
idx2=idx;
for k=1:numDims
  if e(k) > 0	% shift right
    idx{k}=e(k)+1:sizeA(k);
    idx2{k}=1:sizeA(k)-e(k);
  else		% shift left
    idx{k}=1:sizeA(k)+e(k);
    idx2{k}=1-e(k):sizeA(k);
  end
end
As(idx{:})=A(idx2{:});


function [e, sizeA, numDimsA, msg] = ParseInputs(A,e)

% default values
sizeA    = size(A);
numDimsA = ndims(A);
msg      = '';

% Make sure that SHIFTSIZE input is a finite, real integer vector
sh        = e(:);
isFinite  = all(isfinite(sh));
nonSparse = all(~issparse(sh));
isInteger = all(isa(sh,'double') & (imag(sh)==0) & (sh==round(sh)));
isVector  = ismatrix(e);%((ndims(e) == 2) && ((size(e,1) == 1) || (size(e,2) == 1)));

if ~(isFinite && isInteger && isVector && nonSparse)
  msg = ['Invalid shift type: ' ...
    'must be a finite, nonsparse, real integer vector.'];
  return;
end

% Make sure the shift vector has the same length as numDimsA.
% The missing shift values are assumed to be 0. The extra
% shift values are ignored when the shift vector is longer
% than numDimsA.
if (numel(e) < numDimsA)
  e(numDimsA) = 0;
elseif (numel(e) > numDimsA)
  e(numDimsA+1:numel(e))=[];
end

