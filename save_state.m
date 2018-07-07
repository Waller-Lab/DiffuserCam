function out_file = save_state(solverSettings,current_iter)
% Prepares strings for us in MATLAB 'save' function
% Inputs
% solverSettings: settings structure. Must contain field 'save_dir' 
% current_iter: current iteration number
out_file = [solverSettings.save_dir,'/state_',num2str(current_iter),'_',...
    solverSettings.dtstamp,'.mat'];
