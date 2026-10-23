%% ================================================================
%  QUET MUC NHIEU GAUSSIAN (chan doan): LMS co bat dau co ich khi sigma lon hon?
%  So 4 cau hinh (mien van toc): Noisy / Hampel only / Hampel->LMS / Hampel->LMS(WT)
%  Chi tieu quyet dinh: NRMSE cua mu. Neu Hampel->LMS < Hampel-only ro ret o sigma nao
%  => LMS CO ich o che do do => cuu duoc novelty.
%  Yeu cau: LMS_filter_wt.m cung thu muc.
% ================================================================
clc; clear; close all;
addpath(fileparts(mfilename('fullpath')));
rng(2025);
nRuns = 5;                                   % moi muc sigma chay 5 lan
sigma_list = [1.2e-6, 1e-5, 3e-5, 5e-5, 1e-4];   % cac muc nhieu Gaussian can thu

% ---- tham so co dinh ----
Nx=80; Ny=80; dx=1e-3; dy=1e-3; dt=8e-5; nSteps=4000;
frq=100; w=2*pi*frq; rho=1000; A=0.005;
p_impulse=0.002; impulse_amplitude=0.001;
k_hampel=5; threshold_hampel=5; cfg_lms=struct('order',2,'mu',0.2);

% ---- moi truong + khoi u ----
mu_true=6000*ones(Nx,Ny); eta_true=1.2*ones(Nx,Ny);
cx=round(Nx/2); cy=round(Ny/2); radius=15;
for i=1:Nx, for j=1:Ny
    if (i-cx)^2+(j-cy)^2<=radius^2, mu_true(i,j)=9000; eta_true(i,j)=1.8; end
end, end
mu_true(end,end)=10000; mu_true(end,end-1)=0; eta_true(end,end)=2; eta_true(end,end-1)=0;

% ---- FDTD 1 lan ----
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
names={'Noisy','Hampel only','Hampel->LMS','Hampel->LMS(WT)'};
nC=4; nS=numel(sigma_list);
MUn=zeros(nS,nC); MUs=zeros(nS,nC); ETn=zeros(nS,nC); ETs=zeros(nS,nC);

for s=1:nS
    sg=sigma_list(s);
    muR=zeros(nRuns,nC); etR=zeros(nRuns,nC);
    for r=1:nRuns
        fprintf('sigma=%.1e  run %d/%d\n', sg, r, nRuns);
        ng=sg*randn(nSteps,Nx,Ny); ni=zeros(nSteps,Nx,Ny);
        for i=1:Nx, for j=1:Ny
            idx=rand(nSteps,1)<p_impulse; ni(idx,i,j)=impulse_amplitude*(2*randi([0,1],sum(idx),1)-1);
        end, end
        nt=ng+ni; vzn=vz_time+nt;
        vzh=zeros(nSteps,Nx,Ny);
        for i=1:Nx, for j=1:Ny, vzh(:,i,j)=hampel(squeeze(vzn(:,i,j)),k_hampel,threshold_hampel); end, end
        vhl=zeros(nSteps,Nx,Ny);
        for i=1:Nx, for j=1:Ny, vhl(:,i,j)=LMS_filter_wt(squeeze(vzh(:,i,j)),squeeze(nt(:,i,j)),[],cfg_lms); end, end
        vwt=zeros(nSteps,Nx,Ny);
        for i=1:Nx, wp=[]; for j=1:Ny, [o,wp]=LMS_filter_wt(squeeze(vzh(:,i,j)),squeeze(nt(:,i,j)),wp,cfg_lms); vwt(:,i,j)=o; end, end
        [muN,etN]=AHI(vzn,idxf,matran,rho,w,dx);
        [muH,etH]=AHI(vzh,idxf,matran,rho,w,dx);
        [muHL,etHL]=AHI(vhl,idxf,matran,rho,w,dx);
        [muW,etW]=AHI(vwt,idxf,matran,rho,w,dx);
        muMaps={muN,muH,muHL,muW}; etMaps={etN,etH,etHL,etW};
        for c=1:nC
            muR(r,c)=nrmse(mu_true,muMaps{c},mask);
            etR(r,c)=nrmse(eta_true,etMaps{c},mask);
        end
    end
    MUn(s,:)=mean(muR,1); MUs(s,:)=std(muR,0,1);
    ETn(s,:)=mean(etR,1); ETs(s,:)=std(etR,0,1);
end

%% ---- In ket qua ----
fprintf('\n=========== NRMSE(mu) theo sigma_G (mean +/- SD) ===========\n');
fprintf('%-10s %16s %16s %16s %16s\n','sigma_G',names{:});
for s=1:nS
    fprintf('%-10.1e %7.3f±%-7.3f %7.3f±%-7.3f %7.3f±%-7.3f %7.3f±%-7.3f\n', sigma_list(s), ...
        MUn(s,1),MUs(s,1), MUn(s,2),MUs(s,2), MUn(s,3),MUs(s,3), MUn(s,4),MUs(s,4));
end
fprintf('\n=== Cai thien cua Hampel->LMS so voi Hampel-only (mu), %% ===\n');
for s=1:nS
    imp=(MUn(s,2)-MUn(s,3))/MUn(s,2)*100;
    verdict = ''; if imp>5, verdict=' <== LMS CO ich'; end
    fprintf('sigma=%.1e : Hampel=%.3f  H->LMS=%.3f  cai thien=%+.1f%%%s\n', ...
        sigma_list(s), MUn(s,2), MUn(s,3), imp, verdict);
end
fprintf('\n=========== NRMSE(eta) theo sigma_G (mean) ===========\n');
fprintf('%-10s %12s %12s %12s %12s\n','sigma_G',names{:});
for s=1:nS
    fprintf('%-10.1e %12.3f %12.3f %12.3f %12.3f\n', sigma_list(s), ETn(s,1),ETn(s,2),ETn(s,3),ETn(s,4));
end

%% ===== ham phu =====
function [mu_est,eta_est]=AHI(v,idxf,matran,rho,w,dx)
    [~,Nx,Ny]=size(v); Vw=zeros(Nx,Ny);
    for i=1:Nx, for j=1:Ny, Vw(i,j)=matran(idxf,:)*squeeze(v(:,i,j)); end, end
    Lap=4*del2(Vw)/(dx^2); G=(-rho*w^2*Vw)./Lap; mu_est=real(G); eta_est=imag(G)/w;
    mu_est(end,end)=10000; mu_est(end,end-1)=0; eta_est(end,end)=2; eta_est(end,end-1)=0;
end
function v=nrmse(T,E,mask)
    t=double(T(mask)); e=double(E(mask)); g=isfinite(e); t=t(g); e=e(g);
    v=sqrt(mean((t-e).^2))/(max(t)-min(t));
end
