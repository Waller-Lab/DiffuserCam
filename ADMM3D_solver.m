function [vk, f] = ADMM3D_solver(psf,b,solverSettings)
% ADMM solver to compute 3D diffusercam images
% H: Impulse stack 3D array
% b: measurement image from camera
% solverSettings: user defined params. See DiffuserCam_settings.m for
% details

assert(size(psf,1) == size(b,1) || size(psf,2) == size(b,2),'image and impulse have different dimensions');

if ~isfield(solverSettings,'print_interval')
    solverSettings.print_interval = 1;
end

if ~isfield(solverSettings,'disp_figs')
    solverSettings.disp_figs = 1;
end

if ~isfield(solverSettings,'autotune')
    solverSettings.autotune = 1;
end

if ~isfield(solverSettings,'maxIter')
    solverSettings.maxIter = 200;
end

if ~isfield(solverSettings,'regularizer')
    solverSettings.regularizer = 'TV';
end

if ~isfield(solverSettings,'display_func')
    solverSettings.display_func = @(x)x;
end

if ~isfield(solverSettings,'cmap')
    solverSettings.cmap = 'gray';
end

if isfield(solverSettings,'save_vars')
    assert(iscell(solverSettings.save_vars),'solverSettings.save_vars must be a cell array of strings')
else
    solverSettings.save_vars = {'vk'};
end

if strcmpi(solverSettings.save_vars,'')
    solverSettings.save_vars = {'vk'};
end

mu1 = solverSettings.mu1;   %Set initial ADMM parameters
mu2 = solverSettings.mu2;
mu3 = solverSettings.mu3;

[Ny, Nx, Nz] = size(psf);   %Get problem size

% Setup convolutional forward op
p1 = floor(Ny/2);
p2 = floor(Nx/2);
%h(p1,p2,Nz/2) = 1;
pad2d = @(x)padarray(x,[p1,p2],'both');  %2D padding
crop2d = @(x)x(p1+1:end-p1,p2+1:end-p2,:); %2D cropping
crop3d = @(x)crop2d(x(:,:,1));   %3D cropping. This is D
vec = @(X)reshape(X,numel(X),1);
pad3d = @(x)padarray(pad2d(x),[0 0 Nz-1],'post');
psf = circshift(flip(psf,3),ceil(Nz/2)+1,3)/norm(psf(:));  %Shift impulse stack and normalize
Hs = fftn(ifftshift(pad2d(psf)));  %Compute 3D spectrum
Hs_conj = conj(Hs);
clear psf
Hfor = @(x)real((ifftn(Hs.*fftn((x)))));
Hadj = @(x)real((ifftn(Hs_conj.*fftn((x)))));
HtH = abs(Hs.*Hs_conj);


vk = 0*real(Hs);   %Initialize variables. vk is the primal (this is the image you want to find)
xi = vk;  % Dual associated with Mv = nu (boundary condition variables)
rho = vk;  % Dual associated with v = w   (nonnegativity)
Dtb = pad3d(b);

switch lower(solverSettings.regularizer)
    case('tv')
        PsiTPsi = generate_laplacian(vk);
        eta_1 = vk(1:end-1,:,:);  %Duals associatd with Psi v = u (TV sparsity)
        eta_2 = vk(:,1:end-1,:);   %zeros(2*Ny,2*Nx-1,Nz);
        eta_3 = vk(:,:,1:end-1);   %zeros(2*Ny,2*Nx,Nz-1);
        PsiT = @(P1,P2,P3)cat(1,P1(1,:,:),diff(P1,1,1),-P1(end,:,:)) + ...
            cat(2,P2(:,1,:),diff(P2,1,2),-P2(:,end,:)) + ...
            cat(3,P3(:,:,1),diff(P3,1,3),-P3(:,:,end));
       
        % Sparsifying map
        Psi = @(x)deal(-diff(x,1,1),-diff(x,1,2),-diff(x,1,3));
        [uk1, uk2, uk3] = Psi(vk);
        Lvk1 = uk1;
        Lvk2 = uk2;
        Lvk3 = uk3;
    case('tv_native')
        PsiTPsi = generate_laplacian(vk);
        PsiT = @(P1,P2,P3,P4)cat(1,P1(1,:,:),diff(P1,1,1),-P1(end,:,:)) + ...
            cat(2,P2(:,1,:),diff(P2,1,2),-P2(:,end,:)) + ...
            cat(3,P3(:,:,1),diff(P3,1,3),-P3(:,:,end)) + ...
            solverSettings.tau_n/solverSettings.tau*P4;
        
        
        %Sparsifying with gradient and l1
        Psi = @(x)deal(-diff(x,1,1),-diff(x,1,2),-diff(x,1,3),...
            solverSettings.tau_n/solverSettings.tau*x);
        [uk1, uk2, uk3, uk4] = Psi(vk);
        Lvk1 = uk1;
        Lvk2 = uk2;
        Lvk3 = uk3;
        Lvk4 = uk4;
        eta_1 = vk(1:end-1,:,:);  %Duals associatd with Psi v = u (TV sparsity)
        eta_2 = vk(:,1:end-1,:);   %zeros(2*Ny,2*Nx-1,Nz);
        eta_3 = vk(:,:,1:end-1);   %zeros(2*Ny,2*Nx,Nz-1);
        eta_4 = vk;
        PsiTPsi = PsiTPsi + solverSettings.tau_n^2/solverSettings.tau^2;
    case('native')
        
        PsiTPsi = 1;
        PsiT = @(x)x;
        Psi = @(x)x;   %Identity operator for native sparsity
        uk = vk;
        Lvk = uk;
        eta = uk;
end

v_mult = 1./(mu1*HtH + mu2*PsiTPsi + mu3);  %Denominator of v update (in 3D frequency space)

DtD = pad3d(ones(Ny, Nx, 'like', b)); % Initialize DtD with same datatype as input
nu_mult = 1./(DtD + mu1);   %denominator of nu update

n = 0;  %Initialize number of steps to 0

% Store solver parameters in structure, f
% Initialize residuals with NaNs
f.dual_resid_s = zeros(1,solverSettings.maxIter)./0;   
f.primal_resid_s = zeros(1,solverSettings.maxIter)./0;
f.dual_resid_u = f.dual_resid_s;
f.primal_resid_u = f.dual_resid_u;
f.dual_resid_w = f.dual_resid_s;
f.primal_resid_w = f.dual_resid_s;
f.objective = f.primal_resid_u;   
f.data_fidelity = f.primal_resid_u;
f.regularizer_penalty = f.primal_resid_u;
Hvkp = vk;
tic
while n<solverSettings.maxIter
    n = n+1;
    Hvk = Hvkp;
    nukp = nu_mult.*(mu1*(xi/mu1 + Hvk) + Dtb);
    wkp = max(rho/mu3 + vk,0);
    switch lower(solverSettings.regularizer)
        case('tv')
            [uk1, uk2, uk3] = DiffuserCam_soft_3d(Lvk1+eta_1/mu2, Lvk2+eta_2/mu2, Lvk3+eta_3/mu2,solverSettings.tau/mu2);
            vkp_numerator = mu3*(wkp-rho/mu3) + ...
                mu2*PsiT(uk1 - eta_1/mu2,uk2 - eta_2/mu2, uk3 - eta_3/mu2) + ...
                mu1*Hadj(nukp - xi/mu1);
        case('tv_native')
            [uk1, uk2, uk3, uk4] = DiffuserCam_soft_3d(Lvk1 + eta_1/mu2, Lvk2 + eta_2/mu2, ...
                Lvk3 + eta_3/mu2, solverSettings.tau/mu2, Lvk4 + eta_4/mu2);
            vkp_numerator = mu3*(wkp-rho/mu3) + ...
                mu2*PsiT(uk1 - eta_1/mu2,uk2 - eta_2/mu2, uk3 - eta_3/mu2, uk4 - eta_4/mu2) + ...
                mu1*Hadj(nukp - xi/mu1);
        case('native')
            uk = DiffuserCam_soft_3d([],[],[],solverSettings.tau_n/mu2,Lvk + eta/mu2);
            vkp_numerator = mu3*(wkp-rho/mu3) + mu2*PsiT(uk - eta/mu2) + mu1*Hadj(nukp - xi/mu1);
    end
    
    
    vkp = real(ifftn(v_mult .* fftn(vkp_numerator)));
    
    %Update dual and parameter for Hs=v constraint
    Hvkp = Hfor(vkp);
    r_sv = Hvkp-nukp;
    xi = xi + mu1*r_sv;
    f.dual_resid_s(n) = gather(mu1*norm(vec(Hvk - Hvkp)));
    f.primal_resid_s(n) = gather(norm(vec(r_sv)));
    [mu1, mu1_update] = ADMM3D_update_param(mu1,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,f.primal_resid_s(n),f.dual_resid_s(n));
    
    % Update dual and parameter for Ls=v
    f.data_fidelity(n) = gather(.5*norm(crop3d(Hvkp)-b,'fro')^2);
    switch lower(solverSettings.regularizer)
        case('tv')
            Lvk1_ = Lvk1;
            Lvk2_ = Lvk2;
            Lvk3_ = Lvk3;
            [Lvk1, Lvk2, Lvk3] = Psi(vkp);
            r_su_1 = Lvk1 - uk1;
            r_su_2 = Lvk2 - uk2;
            r_su_3 = Lvk3 - uk3;
            eta_1 = eta_1 + mu2*r_su_1;
            eta_2 = eta_2 + mu2*r_su_2;
            eta_3 = eta_3 + mu2*r_su_3;
            f.dual_resid_u(n) = gather(mu2*sqrt(norm(vec(Lvk1_ - Lvk1))^2 + norm(vec(Lvk2_ - Lvk2))^2 + norm(vec(Lvk3_ - Lvk3))^2));
            f.primal_resid_u(n) = gather(sqrt(norm(vec(r_su_1))^2 + norm(vec(r_su_2))^2 + norm(vec(r_su_3))^2));
            f.regularizer_penalty(n) = gather(solverSettings.tau*(sum(vec(abs(Lvk1))) + sum(vec(abs(Lvk2))) + sum(vec(abs(Lvk3)))));
            
        case('tv_native')
            Lvk1_ = Lvk1;
            Lvk2_ = Lvk2;
            Lvk3_ = Lvk3;
            Lvk4_ = Lvk4;
            [Lvk1, Lvk2, Lvk3, Lvk4] = Psi(vkp);
            r_su_1 = Lvk1 - uk1;
            r_su_2 = Lvk2 - uk2;
            r_su_3 = Lvk3 - uk3;
            r_su_4 = Lvk4 - uk4;
            eta_1 = eta_1 + mu2*r_su_1;
            eta_2 = eta_2 + mu2*r_su_2;
            eta_3 = eta_3 + mu2*r_su_3;
            eta_4 = eta_4 + mu2*r_su_4;
            f.dual_resid_u(n) = gather(mu2*sqrt(norm(vec(Lvk1_ - Lvk1))^2 + norm(vec(Lvk2_ - Lvk2))^2 + ...
                norm(vec(Lvk3_ - Lvk3))^2 + norm(vec(Lvk4_ - Lvk4))^2));
            f.primal_resid_u(n) = gather(sqrt(norm(vec(r_su_1))^2 + norm(vec(r_su_2))^2 + ...
                norm(vec(r_su_3))^2 + norm(vec(r_su_4))^2));
           f.regularizer_penalty(n) = gather(solverSettings.tau*(sum(vec(abs(Lvk1))) +...
               sum(vec(abs(Lvk2))) + sum(vec(abs(Lvk3)))) + ...
               solverSettings.tau_n*sum(vec(abs(Lvk4))));
        case('native')
            Lvk_ = Lvk;
            Lvk = Psi(vkp);
            r_su = Lvk - uk;
            eta = eta + mu2*r_su;
            f.dual_resid_u(n) = gather(mu2*norm(vec(Lvk_ - Lvk)));
            f.primal_resid_u(n) = gather(norm(vec(r_su)));
            f.regularizer_penalty(n) = gather(solverSettings.tau_n*(sum(vec(abs(Lvk)))));
    end
    f.objective(n) = f.data_fidelity(n) + f.regularizer_penalty(n);
    
    
    [mu2, mu2_update] = ADMM3D_update_param(mu2,solverSettings.resid_tol,...
        solverSettings.mu_inc,solverSettings.mu_dec,...
        f.primal_resid_u(n),f.dual_resid_u(n));
    
    % Update nonnegativity dual and parameter (s=w)
    r_sw = vkp-wkp;
    rho = rho + mu3*r_sw;
    f.dual_resid_w(n) = gather(mu3*norm(vec(vk - vkp)));
    f.primal_resid_w(n) = gather(norm(vec(r_sw)));
    [mu3, mu3_update] = ADMM3D_update_param(mu3,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,f.primal_resid_w(n),f.dual_resid_w(n));
    
    %Update filters
    if mu1_update || mu2_update || mu3_update
        %fprintf('Mu updates: %i \t %i \t %i\n',mu1_update, mu2_update, mu3_update);
        mu_update = 1;
    else
        mu_update = 0;
    end
    if mu_update
        v_mult = 1./(mu1*HtH + mu2*PsiTPsi + mu3);  %This is the frequency space division fo S update
        nu_mult = 1./(DtD + mu1);
    end
    
    
    vk = vkp;
    
    if mod(n,solverSettings.save_every) == 0
        fprintf('saving state %i...\n',n)
        out_file = save_state(solverSettings,n);
        save(out_file,solverSettings.save_vars{:});
        fprintf('done saving\n')
    end
    
    if mod(n,solverSettings.print_interval) == 0
        t_iter = toc/solverSettings.print_interval;
         fprintf('iter: %i \t t: %.2g \t cost: %.2g \t data_fidelity: %.2g \t norm: %.2g \t Primal v: %.2g \t Dual v: %.2g \t Primal u: %.2g \t Dual u: %.2g \t Primal w: %.2g \t Dual w: %.2g \t mu1: %.2g \t mu2: %.2g \t mu3: %.2g \n',...
            n,t_iter,f.objective(n),f.data_fidelity(n),f.regularizer_penalty(n),f.primal_resid_s(n), f.dual_resid_s(n),f.primal_resid_u(n), f.dual_resid_u(n),f.primal_resid_w(n), f.dual_resid_w(n),mu1,mu2,mu3)
            %disp([n,f.objective(n),f.data_fidelity(n),f.regularizer_penalty(n),f.primal_resid_s(n), f.dual_resid_s(n),f.primal_resid_u(n), f.dual_resid_u(n),f.primal_resid_w(n), f.dual_resid_w(n),mu1,mu2,mu3])
        tic;
    end
    if mod(n,solverSettings.disp_figs) == 0
        draw_figures(vk,solverSettings)
    end
end
end


% Private function to display figures
function draw_figures(xk, solverSettings)
set(0,'CurrentFigure',solverSettings.fighandle)
if numel(size(xk))==2
    imagesc(solverSettings.display_func(xk))
    axis image
    colorbar
    colormap(solverSettings.color_map);
    
elseif numel(size(xk))==3
    xk = solverSettings.disp_crop(xk);
    subplot(1,3,1)
    
    im1 = squeeze(sum(xk,3));
    imagesc(solverSettings.display_func(im1));
    hold on
    axis image
    colormap (solverSettings.cmap)
    %colorbar
    caxis([0 prctile(im1(:),solverSettings.disp_percentile)])
    set(gca,'fontSize',6)
    axis off
    title('XY')
    hold off
    
    subplot(1,3,2)
    im2 = squeeze(max(xk,[],1));
    imagesc(im2);
    hold on    
    %axis image
    colormap (solverSettings.cmap)
    %colorbar
    set(gca,'fontSize',8)
    caxis([0 prctile(im2(:),solverSettings.disp_percentile)])
    title('XZ')
    axis off
    hold off

    
    subplot(1,3,3)
    im3 = squeeze(max(xk,[],2));
    imagesc(solverSettings.disp_func(im3));
    hold on
    %axis image
    colormap (solverSettings.cmap)
    title('YZ')
    colorbar   
    set(gca,'fontSize',8)
    caxis([0 prctile(im3(:),solverSettings.disp_percentile)]);
    axis off
    hold off
    
end

drawnow
end

function PsiTPsi = generate_laplacian(lapl)  %Takes in an array and makes laplacian on same grid (fft shifted)
    %lapl = zeros(2*Ny,2*Nx,Nz);    %Compute laplacian in closed form. This is the kernal to compute Psi'Psi
    lapl(1) = 6;
    lapl(1,2,1) = -1;
    lapl(2,1,1) = -1;
    lapl(1,1,2) = -1;
    lapl(1,end,1) = -1;
    lapl(end,1,1) = -1;
    lapl(1,1,end) = -1;
    PsiTPsi = abs(fftn(lapl));   %Compute power spectrum of laplacian
end

function [mu_out, mu_update] = ADMM3D_update_param(mu,resid_tol,mu_inc,mu_dec,r,s)
    if r > resid_tol*s
        mu_out = mu*mu_inc;
        mu_update = 1;
    elseif r*resid_tol < s
        mu_out = mu/mu_dec;
        mu_update = -1;
    else
        mu_out = mu;
        mu_update = 0;
    end
end
    
    
