function [xhat, f] = DiffuserCam_main(config)
% Solve for image from DiffuserCam. First rev: 3D ADMM only. 
% CONFIG: String with path to settings file. See DiffuserCam_settings.m for
% details.




% Read in settings
eval(config) %This should be a string path to a .m script that populates a bunch of variables in your workspace

%Setup output folder
if solverSettings.save_dir(end) == '/'
    solverSettings.save_dir = solverSettings.save_dir(1:end-1);
end
    
if ~exist(solverSettings.save_dir,'dir') && solverSettings.save_every ~= 0
    mkdir(solverSettings.save_dir);
end

%Make figure handle

if solverSettings.disp_figs
    solverSettings.fighandle = figure(fig_num);
    clf
end

if (lateral_downsample < 1)
    error('lateral_downsample must be >= 1')
end
if (axial_downsample < 1 )
    error('axial_downsample must be >= 1')
end
%Load and prepare impulse stack
psf_in = load(impulse_mat_file_name);
psf = psf_in.(impulse_var_name);

% Get impulse dimensions
[~,~, Nz_in] = size(psf);

if end_z == 0
    end_z = Nz_in;
end

psf = psf(:,:,start_z:end_z);  %Overwrite to save memory

% Do downsampling
for n = 1:lateral_downsample/2
    psf = 1/4*(psf(1:2:end,1:2:end,:)+psf(1:2:end,2:2:end,:) + ...
        psf(2:2:end,1:2:end,:) + psf(2:2:end,2:2:end,:));
end

for n = 1:axial_downsample/2
    psf = 1/2*(psf(:,:,1:2:end)+psf(:,:,2:2:end));
end

[Ny, Nx, Nz] = size(psf);
% Normalize each slice
for n = 1:Nz
    psf(:,:,n) = psf(:,:,n)/norm(psf(:,:,n),'fro');
end

% Load image file and adjust to impulse size.
raw_in = imread(image_file);
colind = find(contains({'red','green','blue'},color_to_process));

if numel(size(image_file)) == 3
    if strcmpi(color_to_process,'mono')
        imc = mean(double(raw_in),3);
    else
        imc = double(raw_in(:,:,colind));
    end
else
    imc = double(raw_in);
end

b = imresize(imc - image_bias,[Ny, Nx],'box');


% Solver stuff


out_file = [solverSettings.save_dir,'/state_',num2str(solverSettings.maxIter)];
if exist(out_file,'file')
    fprintf('file already exists. Adding datetime stamp to avoid overwriting. \n');
    dtstamp = datestr(datetime('now'),'YYYYMMDD_hhmmss');
    out_file = [out_file,'_dtstamp'];
end
xhat = [];
f = [];
save([out_file,'.mat'],'xhat','b','f','raw_in');   %Save result
copyfile(config,solverSettings.save_dir)  %Copy settings into save directory


