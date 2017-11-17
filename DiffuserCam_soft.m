function threshed = DiffuserCam_soft(x,tau)

threshed = max(abs(x)-tau,0);
threshed = threshed.*sign(x);