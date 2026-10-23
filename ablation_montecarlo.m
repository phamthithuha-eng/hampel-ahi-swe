%% ================================================================
%  ABLATION + MONTE-CARLO cho bai RIMNI (B1 + B2 + B3 + bang chung weight-transfer)
%  KHONG doi phuong phap; chi tach dung cac baseline + chay lap lay mean±SD.
%  Yeu cau: file LMS_filter_wt.m cung thu muc; Signal + Image Processing Toolbox.
%
%  7 cau hinh so sanh (cung metric, cung mask):
%    1 Noisy            : khong loc
%    2 Median           : medfilt1 theo thoi gian (mien van toc)
%    3 Wiener           : wiener2 tren ban do tai tao tu Noisy (mien anh)
%    4 LMS only         : LMS (khong Hampel), input = noisy          <-- B1 (LMS THAT)
%    5 Hampel only      : chi Hampel
%    6 Hampel->LMS      : Hampel roi LMS (KHONG truyen trong so)      <-- = bai hien tai
%    7 Hampel->LMS (WT) : Hampel roi LMS CO truyen trong so           <-- novelty R1.9
% ================================================================
clc; clear; close all;
addpath(fileparts(mfilename('fullpath')));
rng(2025);                         % seed co dinh (R1.2)
nRuns = 10;                        % so lan Monte-Carlo (tang len 20-30 neu may du khoe)

% ---------- Tham so (giong Hample_LMS_RIMNI_Final.m) ----------
Nx=80; Ny=80; dx=1e-3; dy=1e-3; dt=8e-5; nSteps=4000;
frq=100; w=2*pi*frq; rho=1000; A=0.005;
sigma_gauss=12e-7; p_impulse=0.002; impulse_amplitude=0.001;
k_hampel=5; threshold_hampel=5;
cfg_lms = struct('order',2,'mu',0.2);

% ---------- Moi truong + khoi u ----------
mu_true=6000*ones(Nx,Ny); eta_true=1.2*ones(Nx,Ny);
cx=round(Nx/2); cy=round(Ny/2); radius=15;
for i=1:Nx, for j=1:Ny
    if (i-cx)^2+(j-cy)^2<=radius^2, mu_true(i,j)=9000; eta_true(i,j)=1.8; end
end, end
mu_true(end,end)=10000; mu_true(end,end-1)=0;     % co dinh thang mau (giong ban goc)
eta_true(end,end)=2;    eta_true(end,end-1)=0;

% ---------- FDTD chay 1 lan -> truong sach ----------
fprintf('FDTD...\n');
vz=zeros(Nx,Ny); vz_time=zeros(nSteps,Nx,Ny); temp=zeros(Nx,Ny);
sgm_zx=zeros(Nx,Ny); sgm_zy=zeros(Nx,Ny);
for t=1:nSteps
    vz(1,1)=A*cos(w*(t-1)*dt);
    vz(1,2)=A*cos(w*(t-1)*dt+dx/(2*pi)); vz(2,1)=vz(1,2); vz(2,2)=vz(1,2);
    for i=2:Nx-1, for j=2:Ny-1
        sgm_zx(i,j)=sgm_zx(i,j)+mu_true(i,j)*dt/dx*(vz(i+1,j)-vz(i,j))+eta_true(i,j)/dx*(vz(i+1,j)-vz(i,j))-eta_true(i,j)/dx*(temp(i+1,j)-temp(i,j));
        sgm_zy(i,j)=sgm_zy(i,j)+mu_true(i,j)*dt/dy*(vz(i,j+1)-vz(i,j))+eta_true(i,j)/dy*(vz(i,j+1)-vz(i,j))-eta_true(i,j)/dy*(temp(i,j+1)-temp(i,j));
        temp(i,j)=vz(i,j);
        vz(i,j)=vz(i,j)+dt/(rho*dx)*(sgm_zx(i,j)-sgm_zx(i-1,j))+dt/(rho*dy)*(sgm_zy(i,j)-sgm_zy(i,j-1));
    end, end
    vz_time(t,:,:)=vz;
end

% ---------- AHI + mask + metrics ----------
L=nSteps; matran=dftmtx(L); fax=(0:L-1)/(L*dt); [~,idxf]=min(abs(fax-frq));
mask=true(Nx,Ny); mask(end,end)=false; mask(end,end-1)=false;
names={'Noisy','Median','Wiener','LMS only','Hampel only','Hampel->LMS','Hampel->LMS (WT)'};
nC=numel(names);
Mu=zeros(nRuns,nC,4);  Et=zeros(nRuns,nC,4);   % [run, config, (NRMSE,PSNR,Corr,SSIM)]

for r=1:nRuns
    fprintf('Run %d/%d...\n', r, nRuns);
    % --- nhieu ---
    ng=sigma_gauss*randn(nSteps,Nx,Ny);
    ni=zeros(nSteps,Nx,Ny);
    for i=1:Nx, for j=1:Ny
        idx=rand(nSteps,1)<p_impulse; ni(idx,i,j)=impulse_amplitude*(2*randi([0,1],sum(idx),1)-1);
    end, end
    noise_total=ng+ni; vz_noisy=vz_time+noise_total;

    % --- Hampel ---
    vz_h=zeros(nSteps,Nx,Ny);
    for i=1:Nx, for j=1:Ny
        vz_h(:,i,j)=hampel(squeeze(vz_noisy(:,i,j)),k_hampel,threshold_hampel);
    end, end

    % --- LMS only (B1): tren noisy, khong Hampel, khong transfer ---
    vz_lmsonly=zeros(nSteps,Nx,Ny);
    for i=1:Nx, for j=1:Ny
        vz_lmsonly(:,i,j)=LMS_filter_wt(squeeze(vz_noisy(:,i,j)),squeeze(noise_total(:,i,j)),[],cfg_lms);
    end, end

    % --- Hampel->LMS (khong transfer) ---
    vz_hl=zeros(nSteps,Nx,Ny);
    for i=1:Nx, for j=1:Ny
        vz_hl(:,i,j)=LMS_filter_wt(squeeze(vz_h(:,i,j)),squeeze(noise_total(:,i,j)),[],cfg_lms);
    end, end

    % --- Hampel->LMS (CO truyen trong so) ---
    vz_hlwt=zeros(nSteps,Nx,Ny);
    for i=1:Nx
        wprev=[];
        for j=1:Ny
            [out,wprev]=LMS_filter_wt(squeeze(vz_h(:,i,j)),squeeze(noise_total(:,i,j)),wprev,cfg_lms);
            vz_hlwt(:,i,j)=out;
        end
    end

    % --- Median theo thoi gian (mien van toc) ---
    vz_med=zeros(nSteps,Nx,Ny);
    for i=1:Nx, for j=1:Ny, vz_med(:,i,j)=medfilt1(squeeze(vz_noisy(:,i,j)),5); end, end

    % --- Tai tao AHI ---
    [muN,etN]  =AHI(vz_noisy,idxf,matran,rho,w,dx);
    [muMed,etMed]=AHI(vz_med,idxf,matran,rho,w,dx);
    [muL,etL]  =AHI(vz_lmsonly,idxf,matran,rho,w,dx);
    [muH,etH]  =AHI(vz_h,idxf,matran,rho,w,dx);
    [muHL,etHL]=AHI(vz_hl,idxf,matran,rho,w,dx);
    [muWT,etWT]=AHI(vz_hlwt,idxf,matran,rho,w,dx);
    % Wiener tren ban do tai tao tu Noisy (mien anh)
    muW=wiener2(muN,[5 5]); etW=wiener2(etN,[5 5]);

    muMaps={muN,muMed,muW,muL,muH,muHL,muWT};
    etMaps={etN,etMed,etW,etL,etH,etHL,etWT};
    for c=1:nC
        Mu(r,c,:)=metrics4(mu_true,muMaps{c},mask);
        Et(r,c,:)=metrics4(eta_true,etMaps{c},mask);
    end
end

%% ---------- In bang mean±SD ----------
printtab('SHEAR MODULUS  mu', names, Mu);
printtab('SHEAR VISCOSITY eta', names, Et);
fprintf('\nGhi chu: Median = medfilt1 thoi gian; Wiener = wiener2 tren ban do tai tao.\n');
fprintf('So sanh (6) vs (7) cho thay tac dung TRUYEN TRONG SO (R1.9).\n');

%% ================= ham phu =================
function [mu_est,eta_est]=AHI(v,idxf,matran,rho,w,dx)
    [~,Nx,Ny]=size(v); Vw=zeros(Nx,Ny);
    for i=1:Nx, for j=1:Ny, Vw(i,j)=matran(idxf,:)*squeeze(v(:,i,j)); end, end
    Lap=4*del2(Vw)/(dx^2); G=(-rho*w^2*Vw)./Lap;
    mu_est=real(G); eta_est=imag(G)/w;
    mu_est(end,end)=10000; mu_est(end,end-1)=0; eta_est(end,end)=2; eta_est(end,end-1)=0;
end

function m=metrics4(T,E,mask)
    t=double(T(mask)); e=double(E(mask)); g=isfinite(e); t=t(g); e=e(g);
    mse=mean((t-e).^2); mx=max(t); mn=min(t);
    nrmse=sqrt(mse)/(mx-mn); psnr=10*log10(mx^2/mse); cc=corr(t,e);
    try, s=ssim(E,T,'DynamicRange',max(T(:))-min(T(:))); catch, s=NaN; end
    m=[nrmse,psnr,cc,s];
end

function printtab(ttl,names,X)
    mu=squeeze(mean(X,1)); sd=squeeze(std(X,0,1));  % nC x 4
    fprintf('\n===== %s =====\n',ttl);
    fprintf('%-20s %14s %14s %14s %12s\n','Config','NRMSE','PSNR(dB)','Corr','SSIM');
    for c=1:numel(names)
        fprintf('%-20s %6.3f±%-6.3f %6.2f±%-5.2f %6.3f±%-5.3f %5.3f±%-5.3f\n', names{c},...
            mu(c,1),sd(c,1), mu(c,2),sd(c,2), mu(c,3),sd(c,3), mu(c,4),sd(c,4));
    end
end
