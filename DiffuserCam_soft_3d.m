function [varargout] =  DiffuserCam_soft_3d(v,h,d,tau,varargin)

%Perform isotropic soft thresholding on volume differences, v, h, and d
%using parameter tau. If a 4th input it added, assume it's the original
%volume and soft threshold that as well (for TV + sparsity).
% for TV+native: pass in 1 more inputs being original volume
if size(v,1) ~= 0  %If no v, h, or d passed in, skip gradient thresholding
    mag = sqrt(cat(1,v,zeros(1,size(v,2),size(v,3))).^2 + ...
        cat(2,h,zeros(size(h,1),1,size(h,3))).^2 + ...
        cat(3,d,zeros(size(d,1),size(d,2),1)).^2);
    magt = DiffuserCam_soft(mag,tau);
    mmult = magt./mag;
    mmult(mag==0) = 0;
    varargout{1} = v.*mmult(1:end-1,:,:);
    varargout{2} = h.*mmult(:,1:end-1,:);
    varargout{3} = d.*mmult(:,:,1:end-1);
    if ~isempty(varargin)  %Fourth argument is native sparsity
        varargout{4} = DiffuserCam_soft(varargin{1},tau);
    end
else
    varargout{1} = DiffuserCam_soft(varargin{1},tau);
end


