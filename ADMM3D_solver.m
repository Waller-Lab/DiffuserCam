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
    solverSettings.display_func = @(x)x
end

mu1 = solverSettings.mu1;   %Set initial ADMM parameters
mu2 = solverSettings.mu2;
mu3 = solverSettings.mu3;

[Ny, Nx, Nz] = size(psf);   %Get problem size


vk = zeros(Ny*2,Nx*2,Nz);   %Initialize variables. vk is the primal (this is the image you want to find)
xi = zeros(2*Ny,2*Nx,Nz);  % Dual associated with Mv = nu (boundary condition variables)
rho = zeros(2*Ny,2*Nx,Nz);  % Dual associated with v = w   (nonnegativity)
Dtb = pad3d(b);

switch lower(solverSettings.regularizer)
    case('tv')
        lapl = zeros(2*Ny,2*Nx,Nz);    %Compute laplacian in closed form. This is the kernal to compute Psi'Psi
        lapl(1) = 6;
        lapl(1,2,1) = -1;
        lapl(2,1,1) = -1;
        lapl(1,1,2) = -1;
        lapl(1,end,1) = -1;
        lapl(end,1,1) = -1;
        lapl(1,1,end) = -1;
        LtL = abs(fftn(lapl));   %Compute power spectrum of laplacian
        eta_1 = zeros(2*Ny-1,2*Nx,Nz);  %Duals associatd with Psi v = u (TV sparsity)
        eta_2 = zeros(2*Ny,2*Nx-1,Nz);
        eta_3 = zeros(2*Ny,2*Nx,Nz-1);
        Ltv3 = @(P1,P2,P3)cat(1,P1(1,:,:),diff(P1,1,1),-P1(end,:,:)) + ...
            cat(2,P2(:,1,:),diff(P2,1,2),-P2(:,end,:)) + ...
            cat(3,P3(:,:,1),diff(P3,1,3),-P3(:,:,end));
       
        % Sparsifying map
        Psi = @(D)deal(-diff(D,1,1),-diff(D,1,2),-diff(D,1,3));
        [uk1, uk2, uk3] = Psi(zeros(2*Ny, 2*Nx,Nz));
        Lvk1 = uk1;
        Lvk2 = uk2;
        Lvk3 = uk3;
    case('tv_native')
        Ltv3 = @(P1,P2,P3,P4)cat(1,P1(1,:,:),diff(P1,1,1),-P1(end,:,:)) + ...
            cat(2,P2(:,1,:),diff(P2,1,2),-P2(:,end,:)) + ...
            cat(3,P3(:,:,1),diff(P3,1,3),-P3(:,:,end)) + ...
            solverSettings.tau_n*P4;
        
        
        %Sparsifying with gradient and l1
        Psi = @(D)deal(-diff(D,1,1),-diff(D,1,2),-diff(D,1,3),solverSettings.tau_n*D);
        [uk1, uk2, uk3, uk4] = Psi(zeros(2*Ny, 2*Nx,Nz));
        Lvk1 = uk1;
        Lvk2 = uk2;
        Lvk3 = uk3;
        Lvk4 = uk4;
        eta_4 = zeros(2*Ny, 2*Nx, Nz);
        LtL = LtL + solverSettings.tau_n^2;
    case('native')
        
        
        Psi = @(x)x;   %Identity operator for native sparsity
        uk = zeros(2*Ny, 2*Nx, Nz);
        Lvk = uk;
        eta = uk;
end

Smult = 1./(mu1*HtH + mu2*LtL + mu3);

DtD = pad3d(ones(size(b)));
Vmult = 1./(DtD + mu1);

n = 0;

dual_resid_s = zeros(1,solverSettings.maxIter)./0;
primal_resid_s = zeros(1,solverSettings.maxIter)./0;
dual_resid_u = dual_resid_s;
primal_resid_u = dual_resid_u;
dual_resid_w = dual_resid_s;
primal_resid_w = dual_resid_s;

f = primal_resid_u;
ps = f;

Hvkp = zeros(2*Ny, 2*Nx,Nz);

while n<solverSettings.maxIter
    n = n+1;
    Hvk = Hvkp;
    nukp = Vmult.*(mu1*(xi/mu1 + Hvk) + Dtb);
    wkp = max(rho/mu3 + vk,0);
    switch lower(solverSettings.regularizer)
        case('tv')
            [uk1, uk2, uk3] = DiffuserCam_soft_3d(Lvk1+eta_1/mu2, Lvk2+eta_2/mu2, Lvk3+eta_3/mu2,solverSettings.tau/mu2);
            vkp_numerator = mu3*(wkp-rho/mu3) + ...
                mu2*Ltv3(uk1 - eta_1/mu2,uk2 - eta_2/mu2, uk3 - eta_3/mu2) + ...
                mu1*Hadj(nukp - xi/mu1);
        case('tv_native')
            [uk1, uk2, uk3, uk4] = DiffuserCam_soft_3d(Lvk1 + eta_1/mu2, Lvk2 + eta_2/mu2, ...
                Lvk3 + eta_3/mu2, solverSettings.tau/mu2, Lvk4 + eta_4/mu2);
            vkp_numerator = mu3*(wkp-rho/mu3) + ...
                mu2*Ltv3(uk1 - eta_1/mu2,uk2 - eta_2/mu2, uk3 - eta_3/mu2, uk4 - eta_4/mu2) + ...
                mu1*Hadj(nukp - xi/mu1);
        case('native')
            uk = DiffuserCam_soft_3d([],[],[],solverSettings.tau_n/mu2,Lvk + eta/mu2);
    end
            
        
    end
    
    
    vkp = real(ifftshift(ifftn(Smult .* fftn(ifftshift(vkp_numerator)))));
    
    %Update dual and parameter for Hs=v constraint
    Hvkp = Hfor(vkp);
    r_sv = Hvkp-nukp;
    xi = xi + mu1*r_sv;
    dual_resid_s(n) = mu1*norm(vec(Hvk - Hvkp));
    primal_resid_s(n) = norm(vec(r_sv));
    [mu1, mu1_update] = update_param(mu1,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,primal_resid_s(n),dual_resid_s(n));
    
    % Update dual and parameter for Ls=v
    
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
            dual_resid_u(n) = mu2*sqrt(norm(vec(Lvk1_ - Lvk1))^2 + norm(vec(Lvk2_ - Lvk2))^2 + norm(vec(Lvk3_ - Lvk3))^2);
            primal_resid_u(n) = sqrt(norm(vec(r_su_1))^2 + norm(vec(r_su_2))^2 + norm(vec(r_su_3))^2);
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
            dual_resid_u(n) = mu2*sqrt(norm(vec(Lvk1_ - Lvk1))^2 + norm(vec(Lvk2_ - Lvk2))^2 + ...
                norm(vec(Lvk3_ - Lvk3))^2 + norm(vec(Lvk4_ - Lvk4))^2);
            primal_resid_u(n) = sqrt(norm(vec(r_su_1))^2 + norm(vec(r_su_2))^2 + ...
                norm(vec(r_su_3))^2 + norm(vec(r_su_4))^2);
        case('native')
            Lvk_ = Lvk;
            Lvk = Psi(vkp);
            r_su = Lvk - uk;
            eta = eta + mu2*r_su;
            dual_resid_u(n) = mu2*norm(vec(Lvk_ - Lvk));
            primal_resid_u(n) = norm(vec(r_su));
    end
    
    
    
    [mu2, mu2_update] = update_param(mu2,solverSettings.resid_tol,...
        solverSettings.mu_inc,solverSettings.mu_dec,...
        primal_resid_u(n),dual_resid_u(n));
    
    % Update nonnegativity dual and parameter (s=w)
    r_sw = vkp-wkp;
    rho = rho + mu3*r_sw;
    dual_resid_w(n) = mu3*norm(vec(vk - vkp));
    primal_resid_w(n) = norm(vec(r_sw));
    [mu3, mu3_update] = update_param(mu3,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,primal_resid_w(n),dual_resid_w(n));
    
    %Update filters
    if mu1_update || mu2_update || mu3_update
        fprintf('Mu updates: %i \t %i \t %i\n',mu1_update, mu2_update, mu3_update);
        mu_update = 1;
    else
        mu_update = 0;
    end
    if mu_update
        Smult = 1./(mu1*HtH + mu2*LtL + mu3);  %This is the frequency space division fo S update
        Vmult = 1./(DtD + mu1);
    end
    
    
    vk = vkp;
    ps(n) = 1/(norm(vec(crop2d(vkp))-vec(imres),1));
    f(n) = norm(crop3d(Hvkp)-b,'fro')^2 + solverSettings.tau*(sum(vec(abs(Lvk1))) + sum(vec(abs(Lvk2))) + sum(vec(abs(Lvk3))) + solverSettings.tau_n*sum(vec(abs(Lvk4))));
    if mod(n,solverSettings.disp_figs)
        draw_figures(vk)
    end
end


% Private function to display figures
function draw_figures(xk)
set(0,'CurrentFigure',solverSettings.fighandle)
if numel(size(xk))==2
    imagesc(solverSettings.display_func(xk))
    axis image
    colorbar
    colormap(solverSettings.color_map);
    
elseif numel(size(xk))==3
    xk = gather(xk);
    subplot(1,3,1)
    
    im1 = squeeze(max(xk,[],3));
    imagesc(solverSettings.display_func(im1));
    hold on
    axis image
    colormap parula
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
    colormap parula
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
    colormap parula
    title('YZ')
    colorbar   
    set(gca,'fontSize',8)
    caxis([0 prctile(im3(:),solverSettings.disp_percentile)]);
    axis off
    hold off
    
end

drawnow
end
    
    
