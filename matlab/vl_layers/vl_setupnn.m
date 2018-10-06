function vl_setupnn()
%VL_SETUPNN Setup the MatConvNet toolbox.
%   VL_SETUPNN() function adds the MatConvNet toolbox to MATLAB path.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

root = vl_rootnn() ;
addpath(fullfile(root, 'matlab'));
addpath(fullfile(root, 'matlab', 'vl_layers'));
addpath(fullfile(root, 'matlab', 'simplenn'));

switch computer
  case 'MACI64'
    os = 'maci';
  case 'GLNXA64'
    [~,cmdout] = system('lsb_release -i');
    idx = strfind(cmdout,':');
    os = lower(cmdout(idx+1:end));
    os(isspace(os))=[];
    case   'PCWIN64'
        os =  'PCWIN64';
  otherwise 
    error('unsupported OS');
end

switch(os)
  case 'maci'
    addpath(fullfile(root, 'matlab', 'mex','maci64'));
  case 'ubuntu'
    addpath(fullfile(root, 'matlab', 'mex','glnxa64'));
    case 'PCWIN64'
        addpath(fullfile(root, 'matlab', 'mex','win64'));
  otherwise 
    error('unsupported OS');    
end

addpath(genpath(fullfile(root, 'matlab','inference')));
addpath(fullfile(root, 'matlab', 'custom_layers'));
addpath(fullfile(root, 'matlab', 'custom_layers','cascades'));
addpath(fullfile(root, 'training'));
