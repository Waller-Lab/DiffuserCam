%impulse_mat_file_name = '~/Documents/randn_psf.mat';
impulse_mat_file_name = './example_data/example_psfs.mat';
impulse_var_name = 'psf';
%image_file = '~/Documents/test_diffusercam.png';  %This will have image_bias subtracted, and be resized to the downsampled impulse stack size using a 'box' filter
image_file = './example_data/example_raw.png';
color_to_process = 'mono';  %'red','green','blue', or 'mono'. If raw file is mono, this is ignored
image_bias = 100;   %If camera has bias, subtract from measurement file. 
lateral_downsample = 1;  %factor to downsample impulse stack laterally. Must be multiple of 2 and >= 1.
axial_downsample = 1;  % Axial averageing of impulse stack. Must be multiple of 2 and >= 1.
 
% Allow user to use subset of Z. This is computed BEFORE downsampling by a
% factor of AXIAL_DOWNSAMPLE
start_z = 1;  %First plane to reconstruct. 1 indexed, as is tradition.
end_z = 0;   %Last plane to reconstruct. If set to 0, use last plane in file.
 
 
% Populate solver options
 
% Solver parameters
solverSettings.tau = .002;    %sparsity parameter for TV
solverSettings.tau_n = 0;     %sparsity param for native sparsity
solverSettings.mu1 = .005;    %Initialize ADMM tuning params. If autotune is on, these will change
solverSettings.mu2 = .05;
solverSettings.mu3 = .5;
 
 
% if set to 1, auto-find mu1, mu2, mu3 every step. If set to 0, use user defined values. If set to N>1, tune for N steps then stop.
solverSettings.autotune = 0;    % default: 1
solverSettings.mu_inc = 1; %1.1;  % 
solverSettings.mu_dec = 1; %1.1;  %Inrement and decrement values for mu during autotune. Turn to 1 to have no tuning.
solverSettings.resid_tol = 1.5;   % Primal/dual gap tolerance. Lower means more frequent tuning
solverSettings.maxIter = 200; % Maximum iteration count  Default: 200
solverSettings.regularizer = 'TV';   %'TV' for 3D TV, 'native' for native. Default: TV
 
%Figures and user info
solverSettings.disp_percentile = 99.9;   %Percentile of max to set image scaling
solverSettings.save_every = 0;   %Save image stack as .mat every N iterations. Use 0 to never save (except for at the end);
if solverSettings.save_every
    warning(' save_every is not enabled yet. Your result will only be saved at the end of processing.')
end
%Folder for saving state. If it doesn't exist, create it. 
solverSettings.save_dir = '~/Documents/Diffusers/Lensless/3d_reconstructions/test';
% Strip / from path if used
 
solverSettings.disp_func = @(x)x;  %Function handle to modify image before display. No change to data, just for display purposes
solverSettings.disp_figs = 1;   %If set to 0, never display. If set to N>=1, show every N.
solverSettings.print_interval = 1;  %Print cost every N iterations. Default 1. If set to 0, don't print.
fig_num = 1;   %Figure number to display in


