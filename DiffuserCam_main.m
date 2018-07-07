function [xhat, f] = DiffuserCam_main(config)
% Solve for image from DiffuserCam. First rev: 3D ADMM only. 
% CONFIG: String with path to settings file. See DiffuserCam_settings.m for
% details.




% Read in settings
run(config); %This should be a string path to a .m script that populates a bunch of variables in your workspace

%Setup output folder
if solverSettings.save_dir(end) == '/'
    solverSettings.save_dir = solverSettings.save_dir(1:end-1);
end
solverSettings.save_dir = [solverSettings.save_dir,'/',solverSettings.dtstamp];
if ~exist(solverSettings.save_dir,'dir')
    mkdir(solverSettings.save_dir);
end

%Make figure handle

if solverSettings.disp_figs ~= 0
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
psf = load(impulse_mat_file_name,impulse_var_name);
psf = psf.(impulse_var_name);

% Get impulse dimensions
[~,~, Nz_in] = size(psf);

if end_z == 0 || end_z > Nz_in
    end_z = Nz_in;
end

psf = psf(:,:,start_z:end_z) - psf_bias;  %Overwrite to save memory

% Do downsampling
for n = 1:log2(lateral_downsample)
    psf = 1/4*(psf(1:2:end,1:2:end,:)+psf(1:2:end,2:2:end,:) + ...
        psf(2:2:end,1:2:end,:) + psf(2:2:end,2:2:end,:));
end

for n = 1:log2(axial_downsample)
    psf = 1/2*(psf(:,:,1:2:end)+psf(:,:,2:2:end));
end

[Ny, Nx, ~] = size(psf);

% Load image file and adjust to impulse size.
raw_in = imread(image_file);
switch color_to_process
    case 'red'; colind = 1;
    case 'green'; colind = 2;
    case 'blue'; colind = 3;
end


if numel(size(image_file)) == 3
    if strcmpi(color_to_process,'mono')
        imc = mean(double(raw_in),3);
    else
        imc = double(raw_in(:,:,colind));
    end
else
    imc = double(raw_in);
end

b = imresize(imc - image_bias,[Ny, Nx],'box'); %Subtract camera bias
b = b/max(b(:));  %Normalize measurement


% Solver stuff

out_file = save_state(solverSettings,solverSettings.maxIter);

if useGpu
    [xhat, f] = ADMM3D_solver(gpuArray(single(psf)),gpuArray(single(b)),solverSettings);
else
    [xhat, f] = ADMM3D_solver(single(psf),single(b),solverSettings);
end
if save_results
    fprintf('saving final results. Please wait. \n')
    xhat_out = gather(xhat);
    save(out_file,'xhat_out','b','f','raw_in');   %Save result
    slashes = strfind(config,'/');
    if ~isempty(slashes)
        config_fname = config(slashes(end)+1:end-2);
    else
        config_fname = config(1:end-2);
    end
    copyfile(config,[solverSettings.save_dir,'/',config_fname,'_',solverSettings.dtstamp,'.m']);  %Copy settings into save directory
    fprintf(['Done. Results saved to ',out_file,'\n'])
end

