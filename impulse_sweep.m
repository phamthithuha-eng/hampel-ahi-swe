%% ================================================================
%  R1.3 - QUET MAT DO & BIEN DO XUNG (robustness). 2 quet 1 chieu:
%    (A) quet mat do p  (bien do A_I co dinh)
%    (B) quet bien do A_I (mat do p co dinh)
%  Chi tieu: NRMSE(mu) cho Noisy / Hampel / Hampel->LMS (mean±SD).
%  Yeu cau: LMS_filter_wt.m cung thu muc.
% ================================================================
clc; clear; close all;
addpath(fileparts(mfilename('fullpath')));
rng(2025); nRuns=5;
Nx=80; Ny=80; dx=1e-3; dy=1e-3; dt=8e-5; nSteps=4000;
frq=100; w=2*pi*frq; rho=1000; A=0.005; sigma_gauss=12e-7;
k_hampel=5; threshold_hampel=5; cfg_lms=struct('order',2,'mu',0.2);

mu_true=6000*ones(Nx,Ny); eta_true=1.2*ones(Nx,Ny);
cx=round(Nx/2); cy=round(Ny/2); radius=15;
for i=1:Nx, for j=1:Ny
    if (i-cx)^2+(j-cy)^2<=radius^2, mu_true(i,j)=9000; eta_true(i,j)=1.8; end
end, end

fprintf('FDTD...\n');
vz=zeros(Nx,Ny); vz_time=zeros(nSteps,Nx,Ny); temp=zeros(Nx,Ny);
sgm_zx=zeros(Nx,Ny); sgm_zy=zeros(Nx,Ny);
for t=1:nSteps
    vz(1,1)=A*cos(w*(t-1)*dt); vz(1,2)=A*cos(w*(t-1)*dt+dx/(2*pi)); vz(2,1)=vz(1,2); vz(2,2)=vz(1,2);
    for i=2:Nx-1, for j=2:Ny-1
        sgm_zx(i,j)=sgm_zx(i,j)+mu_true(i,j)*dt/dx*(vz(i+1,j)-vz(i,j))+eta_true(i,j)/dx*(vz(i+1,j)-vz(i,j))-eta_true(i,j)/dx*(temp(i+1,j)-temp(i,j));
        sgm_zy(i,j)=sgm_zy(i,j)+mu_true(i,j)*dt/dy*(vz(i,j+1)-vz(i,j))+eta_true(i,j)/dy*(vz(i,j+1)-vz(i,j))-eta_true(i,j)/dy*(temp(i,j+1)-temp(i,j));
        temp(i,j)=vz(i,j);
        vz(i,j)=vz(i,j)+dt/(rho*dx)*(sgm_zx(i,j)-sgm_zx(i-1,j))+dt/(rho*dy)*(sgm_zy(i,j)-sgm_zy(i,j-1));
    end, end
    vz_time(t,:,:)=vz;
end
L=nSteps; matran=dftmtx(L); fax=(0:L-1)/(L*dt); [~,idxf]=min(abs(fax-frq));
mask=true(Nx,Ny); mask(end,end)=false; mask(end,end-1)=false;

runSweep=@(p,AI) sweepOne(p,AI,vz_time,nSteps,Nx,Ny,sigma_gauss,k_hampel,threshold_hampel,cfg_lms,idxf,matran,rho,w,dx,mask,mu_true,nRuns);

% (A) quet mat do p
p_list=[0.001 0.002 0.005 0.01]; AI=0.001;
fprintf('\n===== (A) Quet mat do xung p (A_I=%.0e) : NRMSE(mu) mean±SD =====\n',AI);
fprintf('%-8s %16s %16s %16s\n','p','Noisy','Hampel','Hampel->LMS');
for p=p_list
    R=runSweep(p,AI);
    fprintf('%-8.3f %7.3f±%-7.3f %7.3f±%-7.3f %7.3f±%-7.3f\n', p, R(1,1),R(1,2), R(2,1),R(2,2), R(3,1),R(3,2));
end

% (B) quet bien do A_I
A_list=[0.5e-3 1e-3 2e-3]; p=0.002;
fprintf('\n===== (B) Quet bien do xung A_I (p=%.3f) : NRMSE(mu) mean±SD =====\n',p);
fprintf('%-10s %16s %16s %16s\n','A_I','Noisy','Hampel','Hampel->LMS');
for AI=A_list
    R=runSweep(p,AI);
    fprintf('%-10.1e %7.3f±%-7.3f %7.3f±%-7.3f %7.3f±%-7.3f\n', AI, R(1,1),R(1,2), R(2,1),R(2,2), R(3,1),R(3,2));
end

%% ===== ham phu =====
function R=sweepOne(p,AI,vz_time,nSteps,Nx,Ny,sg,kH,thH,cfg,idxf,matran,rho,w,dx,mask,mu_true,nRuns)
    v=zeros(nRuns,3);
    for r=1:nRuns
        ng=sg*randn(nSteps,Nx,Ny); ni=zeros(nSteps,Nx,Ny);
        for i=1:Nx, for j=1:Ny
            idx=rand(nSteps,1)<p; ni(idx,i,j)=AI*(2*randi([0,1],sum(idx),1)-1);
        end, end
        nt=ng+ni; vzn=vz_time+nt;
        vzh=zeros(nSteps,Nx,Ny);
        for i=1:Nx, for j=1:Ny, vzh(:,i,j)=hampel(squeeze(vzn(:,i,j)),kH,thH); end, end
        vhl=zeros(nSteps,Nx,Ny);
        for i=1:Nx, for j=1:Ny, vhl(:,i,j)=LMS_filter_wt(squeeze(vzh(:,i,j)),squeeze(nt(:,i,j)),[],cfg); end, end
        muN=AHIv(vzn,idxf,matran,rho,w,dx);
        muH=AHIv(vzh,idxf,matran,rho,w,dx);
        muHL=AHIv(vhl,idxf,matran,rho,w,dx);
        v(r,1)=nr(mu_true,muN,mask); v(r,2)=nr(mu_true,muH,mask); v(r,3)=nr(mu_true,muHL,mask);
    end
    R=[mean(v)' std(v)'];   % 3x2
end
function mu_est=AHIv(vf,idxf,matran,rho,w,dx)
    [~,Nx,Ny]=size(vf); Vw=zeros(Nx,Ny);
    for i=1:Nx, for j=1:Ny, Vw(i,j)=matran(idxf,:)*squeeze(vf(:,i,j)); end, end
    Lap=4*del2(Vw)/(dx^2); G=(-rho*w^2*Vw)./Lap; mu_est=real(G);
    mu_est(end,end)=10000; mu_est(end,end-1)=0;
end
function v=nr(T,E,mask)
    t=double(T(mask)); e=double(E(mask)); g=isfinite(e); t=t(g); e=e(g);
    v=sqrt(mean((t-e).^2))/(max(t)-min(t));
end
