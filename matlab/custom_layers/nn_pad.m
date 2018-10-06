function y = nn_pad(x, padSize, dzdy, varargin)
%NN_PAD CNN padding.
%   Y = NN_PAD(X, PADSIZE) pads symmetrically the spatial dimensions of
%   the input X.
%   PADSIZE specifies the amount of padding as [TOP, BOTTOM, LEFT, RIGHT].
%
%   DZDX = NN_PAD(X, PADSIZE, DZDY) computes the derivative DZDX of the
%   function projected on the output derivative DZDY. DZDX has the same
%   dimension as X and DZDY the same dimension as Y.
%
%   DZDX = NN_PAD([], PADSIZE, DZDY) is an alternative to
%   the previous call in which X is omitted.

% stamatis@math.ucla.edu, 06/01/2015

opts.padType = 'symmetric';
opts = vl_argparse(opts,varargin);

assert(ismember(opts.padType,{'symmetric','zero'}),'Unknown boundary extension.');

if numel(padSize)==1
  padSize=padSize*ones(1,4);
end

if ~isempty(x)
  sz=size(x);
elseif ~isempty(dzdy)
  sz=size(dzdy);
  sz(1)=sz(1)-padSize(1)-padSize(2);
  sz(2)=sz(2)-padSize(3)-padSize(4);
else
  sz=[];
end

if padSize(1) > sz(1) || padSize(2) > sz(1) || padSize(3) > sz(2) || padSize(4) > sz(2)
  error('nn_pad:InvalidInput','padSize cannot be greater than inputSize.');
end

if nargin <= 2 || isempty(dzdy)
  
  if isequal(opts.padType,'zero')
    opts.padType = 0;
  end
  
  sflag=false; % Check for equal-size padding on TOP-BOTTOM and LEFT-RIGHT
  if padSize(1)==padSize(2) && padSize(3)==padSize(4)
    sflag=true;
  end
  
  if sflag
    y=padarray(x,[padSize(1),padSize(3)],opts.padType,'both');
  else
    y = padarray(x,[padSize(1),padSize(3)],opts.padType,'pre');
    y = padarray(y,[padSize(2),padSize(4)],opts.padType,'post');
  end
else
  if isequal(opts.padType,'symmetric')
    dzdy(padSize(1)+1:2*padSize(1),:,:,:)=...
      dzdy(padSize(1)+1:2*padSize(1),:,:,:)+dzdy(padSize(1):-1:1,:,:,:);
    
    dzdy(end-2*padSize(2)+1:end-padSize(2),:,:,:)=...
      dzdy(end-2*padSize(2)+1:end-padSize(2),:,:,:)...
      +dzdy(end:-1:end-padSize(2)+1,:,:,:);
    
    dzdy(:,padSize(3)+1:2*padSize(3),:,:)=...
      dzdy(:,padSize(3)+1:2*padSize(3),:,:)+dzdy(:,padSize(3):-1:1,:,:);
    
    dzdy(:,end-2*padSize(4)+1:end-padSize(4),:,:)=...
      dzdy(:,end-2*padSize(4)+1:end-padSize(4),:,:)...
      +dzdy(:,end:-1:end-padSize(4)+1,:,:);
  end
  y=dzdy(padSize(1)+1:end-padSize(2),padSize(3)+1:end-padSize(4),:,:);  
end
