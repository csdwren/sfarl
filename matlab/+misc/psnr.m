function [snr_db, MSE] = psnr(X1, X2, peak)

% ========================== INPUT PARAMETERS (required) ==================
% Parameters    Values and description
% =========================================================================
% X1            processed image-stack.
% X2            ground-truth image-stack.
% peak          peak value used in the computation of the PSNR. (Default:
%               maximum value of X2).
% ========================== OUTPUT PARAMETERS ============================
% score         Modified version of the normalized mean integrated squared 
%               error, which is appropriate for the more general deblurring
%               problem.
% =========================================================================
% =========================================================================
%
% Author: stamatis.lefkimmiatis@epfl.ch
%
% =========================================================================
 
% PSNR(db) = 10*log10( peak^2 / MSE )


if nargin<3, peak=max(abs(X2(:))); end
if ~isequal(size(X1), size(X2)), error('non-matching X1-X2'); end
N = numel(X1);

if N==0
  snr_db=nan;
  MSE=0;
  return;
end

MSE = norm(X1(:)-X2(:),2)^2/N;

if MSE==0
  snr_db=inf;
else
  snr_db = 10*log10(peak^2/MSE);
end