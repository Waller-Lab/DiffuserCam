%impulse_mat_file_name = '~/Documents/randn_psf.mat';
impulse_mat_file_name = '/Users/nick.antipa/Documents/Diffusers/Lensless/3D_calibration/zstackPCO_Color_3D.mat';
impulse_var_name = 'zstackg';
%image_file = '~/Documents/test_diffusercam.png';  %This will have image_bias subtracted, and be resized to the downsampled impulse stack size using a 'box' filter
image_file = '/Users/nick.antipa/Documents/usaf_tilt_reverse.png';
color_to_process = 'red';  %'red','green','blue', or 'mono'. If raw file is mono, this is ignored
image_bias = 100;   %If camera has bias, subtract from measurement file. 
lateral_downsample = 4;  %factor to downsample impulse stack laterally. >1. Must be multiple of 2.
axial_downsample = 2;  % Axial averageing of impulse stack. Must be multiple of 2.

% Allow user to use subset of Z. This is computed BEFORE downsampling by a
% factor of AXIAL_DOWNSAMPLE
start_z = 73;  %First plane to reconstruct. 1 indexed, as is tradition.
end_z = 0;   %Last plane to reconstruct. If set to 0, use last plane in file.


% Populate solver options

% Solver parameters
solverSettings.tau = .2;    %sparsity parameter for TV
solverSettings.tau_n = .2;  %sparsity param for native sparsity
solverSettings.mu1 = .5;    %Initialize ADMM tuning params. If autotune is on, these will change
solverSettings.mu2 = .5;
solverSettings.mu3 = .5;


% if set to 1, auto-find mu1, mu2, mu3 every step. If set to 0, use user defined values. If set to N>1, tune for N steps then stop.
solverSettings.autotune = 1;    % default: 1
solverSettings.mu_inc = 1.1;   %
solverSettings.mu_dec = 1.1;  %Inrement and decrement values for mu during autotune.
solverSettings.resid_tol = 1.5;   % Primal/dual gap tolerance. Lower means more frequent tuning
solverSettings.maxIter = 2000; % Maximum iteration count  Default: 200
solverSettings.regularizer = 'tv';   %'TV' for 3D TV, 'native' for native. Default: TV

%Figures and user info
solverSettings.disp_percentile = 99.9;   %Percentile of max to set image scaling
solverSettings.save_every = 50;   %Save image stack as .mat every N iterations. Use 0 to never save (except for at the end);

%Folder for saving state. If it doesn't exist, create it. 
solverSettings.save_dir = '~/Documents/Diffusers/Lensless/3d_reconstructions/test';
% Strip / from path if used

solverSettings.disp_func = @(x)x;  %Function handle to modify image before display. No change to data, just for display purposes
solverSettings.disp_figs = 1;   %If set to 0, never display. If set to N>=1, show every N.
solverSettings.print_interval = 1;  %Print cost every N iterations. Default 1. If set to 0, don't print.
fig_num = 1;   %Figure number to display in
