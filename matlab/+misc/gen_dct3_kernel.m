function [h1,h2,h3]=gen_dct3_kernel(support,varargin)
% support: array with the spatial support of the filters, e.g support=[5,5,3]
%
% If three output arguments are returned then:
% h1 has dimensions [Px 1 1 Kx], h2 has dimensions [1 Py 1 Ky] and h3 has 
% dimensions [1 1 Pz Kz] where Kx, Ky and Kz are the number of filters and 
% Px, Py and Pz are their support. In this case Kx=Px, Ky=Py and Kz = Pz 
% but we could also define an overcomplete version for dct3.
%
% If two output arguments are returned then:
% h1 has dimensions [Px 1 1 Kx] and h2 has dimensions [1 Py 1 Ky] where
% Kx, Ky are the number of filters and Px, Py are their support. In this
% case Kx=Px and Ky=Py but we could also define an overcomplete version for
% dct2.
%
% Otherwise, h2 is an empty array and h1 has dimensions [Px Py Pz K] where  
% Px, Py and Pz is the support of the filter and K = Px*Py*Pz. (We could 
% also define an overcomplete dct3 kernel where in this case K > Px*Py.

opts.classType='single';
opts.gpu=false;
opts=vl_argparse(opts,varargin);

if numel(support) < 3
  support = [support(:)', ones(1,3-numel(support))];
end

if nargout == 3
  D = cast(dctmtx(support(1)),opts.classType);
  h1 = reshape(D',[support(1) 1 1 support(1)]);
  if support(2) ~= support(1)
    D = cast(dctmtx(support(2)),opts.classType);
  end
  h2 = reshape(D',[1 support(2) 1 support(2)]);
  if support(3) ~= support(2)
    D = cast(dctmtx(support(3)),opts.classType);
  end
  h3 = reshape(D',[1 1 support(3) support(3)]);  
  
  if opts.gpu
    h1 = gpuArray(h1);
    h2 = gpuArray(h2);
    h3 = gpuArray(h3);
  end
elseif nargout == 2
  if support(1) ~= support(2)
    D = cast(dctmtx(support(1)),opts.classType);
    h1 = reshape(D',[support(1) 1 1 support(1)]);
    D = cast(dctmtx(support(2)),opts.classType);
    h2 = reshape(D',[1 support(2) 1 support(2)]);
  else
    D = cast(dctmtx(support(1)),opts.classType);
    h1 = reshape(D',[support(1) 1 1 support(1)]);
    h2 = reshape(D',[1 support(1) 1 support(1)]);
  end
  
  if opts.gpu
    h1 = gpuArray(h1);
    h2 = gpuArray(h2);
  end
  h3 = [];
else
  if numel(support) < 2, support(2) = support(1); end
  if numel(support) < 3, support(3) = 1; end
  
  h1 = zeros(support(1),support(2),support(3),prod(support(1:3)),opts.classType);
  dirac = zeros(support(1),support(2),support(3),opts.classType);
  vec = @(x)x(:);
  for k = 1:support(1)
    for l = 1:support(2)
      for m = 1:support(3)
        dirac(k,l,m) = 1;
        h1(k,l,m,:) = vec(dct3(dirac));
        dirac(k,l,m) = 0;
      end
    end
  end
  
  if opts.gpu
    h1 = gpuArray(h1);
  end
  h2 = [];
  h3 = [];
end
  

function z=dct3(x)

z = zeros(size(x),'like',x);

for c=1:size(x,3)
  z(:,:,c) = dct2(x(:,:,c));
end

for k=1:size(x,1)
  for l=1:size(x,2)
    z(k,l,:) = dct(squeeze(z(k,l,:)));
  end
end




