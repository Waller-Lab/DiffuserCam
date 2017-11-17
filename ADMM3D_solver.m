function [sk, f] = ADMM3D_solver(psf,b,solverSettings)
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


sk = zeros(Ny*2,Nx*2,Nz);   %Initialize variables. sk is the primal (this is the image you want to find)
xi = zeros(2*Ny,2*Nx,Nz);  % Dual associated with Mv = nu (boundary condition variables)
rho = zeros(2*Ny,2*Nx,Nz);  % Dual associated with v = w   (nonnegativity)
Dtb = pad3d(b);

switch lower(sparsity_type)
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
        
        % Forward finite difference (returns two images!) [P1, P2] = L3(x)
        L3 = @(D)deal(-diff(D,1,1),-diff(D,1,2),-diff(D,1,3));
        [uk1, uk2, uk3] = L3(zeros(2*Ny, 2*Nx,Nz));
        Lsk1 = uk1;
        Lsk2 = uk2;
        Lsk3 = uk3;
    case('tv_native')
        Ltv3 = @(P1,P2,P3,P4)cat(1,P1(1,:,:),diff(P1,1,1),-P1(end,:,:)) + ...
            cat(2,P2(:,1,:),diff(P2,1,2),-P2(:,end,:)) + ...
            cat(3,P3(:,:,1),diff(P3,1,3),-P3(:,:,end)) + ...
            solverSettings.tau_n*P4;
        L3 = @(D)deal(-diff(D,1,1),-diff(D,1,2),-diff(D,1,3),solverSettings.tau_n*D);
        [uk1, uk2, uk3, uk4] = L3(zeros(2*Ny, 2*Nx,Nz));
        Lsk1 = uk1;
        Lsk2 = uk2;
        Lsk3 = uk3;
        Lsk4 = uk4;
        eta_4 = zeros(2*Ny, 2*Nx, Nz);
        LtL = LtL + solverSettings.tau_n^2;
    case('native')
        eta = zeros(2*Ny, 2*Nx, Nz);
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

Hskp = zeros(2*Ny, 2*Nx,Nz);

while n<solverSettings.maxIter
    n = n+1;
    
    %[Lsk1, Lsk2, Lsk3] = L3(sk);
    
    %uk1_ = uk1;
    %uk2_ = uk2;
    %uk3_ = uk3;
    Hsk = Hskp;
    vkp = Vmult.*(mu1*(xi/mu1 + Hsk) + Dtb);
    wkp = max(rho/mu3 + sk,0);
    switch lower(sparsity_type)
        case('tv')
            [uk1, uk2, uk3] = soft_3d_gradient(Lsk1+eta_1/mu2, Lsk2+eta_2/mu2, Lsk3+eta_3/mu2,solverSettings.tau/mu2);
            skp_numerator = mu3*(wkp-rho/mu3) + ...
                mu2*Ltv3(uk1 - eta_1/mu2,uk2 - eta_2/mu2, uk3 - eta_3/mu2) + ...
                mu1*Hadj(vkp - xi/mu1);
        case('tv_native')
            uk4_ = uk4;
            [uk1, uk2, uk3, uk4] = soft_3d_gradient(Lsk1 + eta_1/mu2, Lsk2 + eta_2/mu2, ...
                Lsk3 + eta_3/mu2, solverSettings.tau/mu2, Lsk4 + eta_4/mu2);
            skp_numerator = mu3*(wkp-rho/mu3) + ...
                mu2*Ltv3(uk1 - eta_1/mu2,uk2 - eta_2/mu2, uk3 - eta_3/mu2, uk4 - eta_4/mu2) + ...
                mu1*Hadj(vkp - xi/mu1);
    end
    
    
    skp = real(ifftshift(ifftn(Smult .* fftn(ifftshift(skp_numerator)))));
    
    %Update dual and parameter for Hs=v constraint
    Hskp = Hfor(skp);
    r_sv = Hskp-vkp;
    xi = xi + mu1*r_sv;
    dual_resid_s(n) = mu1*norm(vec(Hsk - Hskp));
    primal_resid_s(n) = norm(vec(r_sv));
    [mu1, mu1_update] = update_param(mu1,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,primal_resid_s(n),dual_resid_s(n));
    
    % Update dual and parameter for Ls=v
    Lsk1_ = Lsk1;
    Lsk2_ = Lsk2;
    Lsk3_ = Lsk3;
    switch lower(sparsity_type)
        case('tv')
            [Lsk1, Lsk2, Lsk3] = L3(skp);
            r_su_1 = Lsk1 - uk1;
            r_su_2 = Lsk2 - uk2;
            r_su_3 = Lsk3 - uk3;
            eta_1 = eta_1 + mu2*r_su_1;
            eta_2 = eta_2 + mu2*r_su_2;
            eta_3 = eta_3 + mu2*r_su_3;
            dual_resid_u(n) = mu2*sqrt(norm(vec(Lsk1_ - Lsk1))^2 + norm(vec(Lsk2_ - Lsk2))^2 + norm(vec(Lsk3_ - Lsk3))^2);
            primal_resid_u(n) = sqrt(norm(vec(r_su_1))^2 + norm(vec(r_su_2))^2 + norm(vec(r_su_3))^2);
        case('tv_native')
            Lsk4_ = Lsk4;
            [Lsk1, Lsk2, Lsk3, Lsk4] = L3(skp);
            r_su_1 = Lsk1 - uk1;
            r_su_2 = Lsk2 - uk2;
            r_su_3 = Lsk3 - uk3;
            r_su_4 = Lsk4 - uk4;
            eta_1 = eta_1 + mu2*r_su_1;
            eta_2 = eta_2 + mu2*r_su_2;
            eta_3 = eta_3 + mu2*r_su_3;
            eta_4 = eta_4 + mu2*r_su_4;
            dual_resid_u(n) = mu2*sqrt(norm(vec(Lsk1_ - Lsk1))^2 + norm(vec(Lsk2_ - Lsk2))^2 + ...
                norm(vec(Lsk3_ - Lsk3))^2 + norm(vec(Lsk4_ - Lsk4))^2);
            primal_resid_u(n) = sqrt(norm(vec(r_su_1))^2 + norm(vec(r_su_2))^2 + ...
                norm(vec(r_su_3))^2 + norm(vec(r_su_4))^2);
    end
    
    
    
    [mu2, mu2_update] = update_param(mu2,solverSettings.resid_tol,solverSettings.mu_inc,solverSettings.mu_dec,primal_resid_u(n),dual_resid_u(n));
    
    % Update nonnegativity dual and parameter (s=w)
    r_sw = skp-wkp;
    rho = rho + mu3*r_sw;
    dual_resid_w(n) = mu3*norm(vec(sk - skp));
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
    
    
    sk = skp;
    ps(n) = 1/(norm(vec(crop2d(skp))-vec(imres),1));
    f(n) = norm(crop3d(Hskp)-b,'fro')^2 + solverSettings.tau*(sum(vec(abs(Lsk1))) + sum(vec(abs(Lsk2))) + sum(vec(abs(Lsk3))) + solverSettings.tau_n*sum(vec(abs(Lsk4))));
    if mod(n,10)==0
        
        fprintf('iter: %i \t cost: %.4f \t Primal v: %.4f \t Dual v: %.4f \t Primal u: %.4f \t Dual u: %.4f \t Primal w: %.4f \t Dual w: %.4f \t mu1: %.4f \t mu2: %.4f \t mu3: %.4f \n',...
            n,f(n),primal_resid_s(n), dual_resid_s(n),primal_resid_u(n), dual_resid_u(n),primal_resid_w(n), dual_resid_w(n),mu1,mu2,mu3)
        
        set(0,'CurrentFigure',h1);
        
        imagesc(crop2d(max(skp,[],3)));
        axis image
        colormap parula
        drawnow
        
        if n > 50
            set(0,'CurrentFigure',h2)
            plot(ps(n-49:n));
        end
        
        
        
    end
end


% Private function to display figures
function draw_figures(xk,options)
set(0,'CurrentFigure',options.fighandle)
if numel(size(xk))==2
    imagesc(options.display_func(xk))
    axis image
    colorbar
    colormap(options.color_map);
    
elseif numel(size(xk))==3
    xk = gather(xk);
    subplot(1,3,1)
    
    im1 = squeeze(max(xk,[],3));
    imagesc(options.display_func(im1));
    hold on
    axis image
    colormap parula
    %colorbar
    caxis([0 prctile(im1(:),options.disp_percentile)])
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
    caxis([0 prctile(im2(:),options.disp_percentile)])
    title('XZ')
    axis off
    hold off

    
    subplot(1,3,3)
    im3 = squeeze(max(xk,[],2));
    imagesc(options.disp_func(im3));
    hold on
    %axis image
    colormap parula
    title('YZ')
    colorbar   
    set(gca,'fontSize',8)
    caxis([0 prctile(im3(:),options.disp_percentile)]);
    axis off
    hold off
    
end

drawnow
end
    
    
