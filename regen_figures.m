%% ================================================================
%  REGEN_FIGURES  (v2026-10-23)  -- tai tao 2 hinh cho ban revision RIMNI
%    figures/figure11.pdf : Fig. 4  histogram nhieu truoc/sau Hampel  (bo "Truncated Gaussian")
%    figures/figure14.pdf : Fig. 8  ban do mu: GT / No filtering / Median / Wiener / Hampel->LMS / MAE
%
%  - Cung pipeline & cung thu tu sinh nhieu nhu ablation_montecarlo.m, rng(2025)
%    => trung voi RUN 1 cua ablation (Bang 4).
%  - Ve off-screen ('Visible','off') de chay duoc tren MATLAB Online.
%  - In ra cac so lieu can dong bo vao bai (mean/std truoc-sau, % giam, MAE tung phuong phap).
%  Yeu cau: LMS_filter_wt.m cung thu muc; Signal + Image Processing Toolbox.
% ================================================================
clc; clear; close all;
addpath(fileparts(mfilename('fullpath')));
outdir = fullfile(fileparts(mfilename('fullpath')),'figures'); if ~exist(outdir,'dir'), mkdir(outdir); end
fprintf('regen_figures v2026-10-23\n');
rng(2025);

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
mu_true(end,end)=10000; mu_true(end,end-1)=0;
eta_true(end,end)=2;    eta_true(end,end-1)=0;

% ---------- FDTD ----------
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

% ---------- Nhieu (CUNG THU TU nhu ablation run 1) ----------
ng=sigma_gauss*randn(nSteps,Nx,Ny);
ni=zeros(nSteps,Nx,Ny);
for i=1:Nx, for j=1:Ny
    idx=rand(nSteps,1)<p_impulse; ni(idx,i,j)=impulse_amplitude*(2*randi([0,1],sum(idx),1)-1);
end, end
noise_total=ng+ni; vz_noisy=vz_time+noise_total;

% ---------- Hampel ----------
fprintf('Hampel...\n');
vz_h=zeros(nSteps,Nx,Ny);
for i=1:Nx, for j=1:Ny
    vz_h(:,i,j)=hampel(squeeze(vz_noisy(:,i,j)),k_hampel,threshold_hampel);
end, end

% ---------- Hampel->LMS (khong truyen trong so = bai hien tai) ----------
fprintf('LMS...\n');
vz_hl=zeros(nSteps,Nx,Ny);
for i=1:Nx, for j=1:Ny
    vz_hl(:,i,j)=LMS_filter_wt(squeeze(vz_h(:,i,j)),squeeze(noise_total(:,i,j)),[],cfg_lms);
end, end

% ---------- Median theo thoi gian ----------
vz_med=zeros(nSteps,Nx,Ny);
for i=1:Nx, for j=1:Ny, vz_med(:,i,j)=medfilt1(squeeze(vz_noisy(:,i,j)),5); end, end

% ---------- AHI ----------
fprintf('AHI...\n');
L=nSteps; matran=dftmtx(L); fax=(0:L-1)/(L*dt); [~,idxf]=min(abs(fax-frq));
[muN ,~]=AHI(vz_noisy,idxf,matran,rho,w,dx);
[muMed,~]=AHI(vz_med ,idxf,matran,rho,w,dx);
[muH ,~]=AHI(vz_h    ,idxf,matran,rho,w,dx);
[muHL,~]=AHI(vz_hl   ,idxf,matran,rho,w,dx);
muW=wiener2(muN,[5 5]);

%% ================= FIG. 4  (figures/figure11.pdf) =================
sig_ideal = squeeze(vz_time(:,cx,cy));   % residual noise at the INCLUSION CENTRE (units: m/s, same as vz)
noise_before = squeeze(vz_noisy(:,cx,cy)) - sig_ideal;
noise_after  = squeeze(vz_h(:,cx,cy))     - sig_ideal;
m1=mean(noise_before); s1=std(noise_before);
m2=mean(noise_after);  s2=std(noise_after);

f4=figure('Visible','off','Units','centimeters','Position',[1 1 24 9],'Color','w');
subplot(1,2,1);
histogram(noise_before,50,'Normalization','pdf','FaceColor',[0.2 0.4 0.8],'EdgeColor','none','FaceAlpha',0.75); hold on;
xv=linspace(min(noise_before),max(noise_before),200);
plot(xv,normpdf(xv,m1,s1),'r-','LineWidth',1.8);
xlabel('Noise amplitude (m/s)'); ylabel('Probability density');
title(sprintf('Before Hampel (heavy-tailed)\nmean = %.3e m/s, std = %.3e m/s',m1,s1),'FontWeight','normal');
legend('Noise histogram','Gaussian fit','Location','northwest'); grid on; box on;
subplot(1,2,2);
histogram(noise_after,50,'Normalization','pdf','FaceColor',[0.2 0.65 0.3],'EdgeColor','none','FaceAlpha',0.75); hold on;
xv=linspace(min(noise_after),max(noise_after),200);
plot(xv,normpdf(xv,m2,s2),'r-','LineWidth',1.8);
xlabel('Noise amplitude (m/s)'); ylabel('Probability density');
title(sprintf('After Hampel (near-Gaussian)\nmean = %.3e m/s, std = %.3e m/s',m2,s2),'FontWeight','normal');
legend('Noise histogram','Gaussian fit','Location','northwest'); grid on; box on;
savefig_pdf(f4, fullfile(outdir,'figure11'));

%% ================= FIG. 8  (figures/figure14.pdf) =================
rr=max(1,cx-radius):min(Nx,cx+radius); cc=max(1,cy-radius):min(Ny,cy+radius);
T=mu_true(rr,cc);
mae=@(M) mean(abs(M(rr,cc)-T),'all','omitnan');
MAE=[mae(muN) mae(muMed) mae(muW) mae(muHL)];
lab={'No filtering','Median','Wiener','Hampel\rightarrowLMS'};

f8=figure('Visible','off','Units','centimeters','Position',[1 1 27 16],'Color','w');
maps={mu_true,muN,muMed,muW,muHL};
ttl={'(a) Ground truth','(b) No filtering (noisy)','(c) Median filter','(d) Wiener filter','(e) Hampel\rightarrowLMS (proposed)'};
clim_=[0 10000];
for p=1:5
    ax=subplot(2,3,p);
    imagesc(maps{p},clim_); axis image; set(ax,'YDir','normal');
    title(ttl{p},'FontWeight','normal');
    xlabel('x (mm)'); ylabel('y (mm)');
    cb=colorbar; cb.Label.String='\mu (Pa)';
end
subplot(2,3,6);
b=bar(MAE,0.6,'FaceColor',[0.3 0.5 0.8]); grid on; box on;
set(gca,'XTickLabel',lab,'XTickLabelRotation',20);
ylabel('Mean absolute error (Pa)'); title('(f) MAE inside inclusion','FontWeight','normal');
for p=1:4, text(p,MAE(p),sprintf('%.0f',MAE(p)),'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',8); end
savefig_pdf(f8, fullfile(outdir,'figure14'));

%% ================= SO LIEU DE DONG BO VAO BAI =================
fprintf('\n===== FIG. 4 (histogram) =====\n');
fprintf('Before: mean=%.3e  std=%.3e\n', m1, s1);
fprintf('After : mean=%.3e  std=%.3e\n', m2, s2);
fprintf('Std reduction      = %.2f %%\n', (1-s2/s1)*100);
fprintf('Variance reduction = %.2f %%\n', (1-(s2/s1)^2)*100);
fprintf('\n===== FIG. 8(f) MAE inside inclusion (Pa) =====\n');
for p=1:4, fprintf('%-18s %10.2f\n', strrep(lab{p},'\rightarrow','->'), MAE(p)); end
fprintf('Relative improvement (No filtering -> Hampel->LMS) = %.2f %%\n', (MAE(1)-MAE(4))/MAE(1)*100);
fprintf('\nFiles: %s\n', outdir);

%% ================= ham phu =================
function [mu_est,eta_est]=AHI(v,idxf,matran,rho,w,dx)
    [~,Nx,Ny]=size(v); Vw=zeros(Nx,Ny);
    for i=1:Nx, for j=1:Ny, Vw(i,j)=matran(idxf,:)*squeeze(v(:,i,j)); end, end
    Lap=4*del2(Vw)/(dx^2); G=(-rho*w^2*Vw)./Lap;
    mu_est=real(G); eta_est=imag(G)/w;
    mu_est(end,end)=10000; mu_est(end,end-1)=0; eta_est(end,end)=2; eta_est(end,end-1)=0;
end
function savefig_pdf(f,base)
    set(f,'PaperPositionMode','auto');
    try
        exportgraphics(f,[base '.pdf'],'ContentType','vector','BackgroundColor','white');
    catch
        try, print(f,[base '.pdf'],'-dpdf','-bestfit'); catch, saveas(f,[base '.pdf']); end
    end
    try, print(f,[base '.png'],'-dpng','-r300'); catch, end
    fprintf('Saved %s.pdf / .png\n', base);
end
