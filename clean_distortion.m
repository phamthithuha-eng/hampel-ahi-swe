%% ================================================================
%  R1.8 - Phan tich DINH LUONG meo tin hieu SACH (khong nhieu)
%  Ap tung bo loc len truong SACH, do: sai lech bien do @100Hz, sai so pha,
%  RMSE dang song, va ti le thay nham mau hop le (false-replacement).
%  Yeu cau: LMS_filter_wt.m cung thu muc.
% ================================================================
clc; clear; close all;
addpath(fileparts(mfilename('fullpath')));
Nx=80; Ny=80; dx=1e-3; dy=1e-3; dt=8e-5; nSteps=4000;
frq=100; w=2*pi*frq; rho=1000; A=0.005;
k_hampel=5; threshold_hampel=5; cfg_lms=struct('order',2,'mu',0.2);

mu_true=6000*ones(Nx,Ny); eta_true=1.2*ones(Nx,Ny);
cx=round(Nx/2); cy=round(Ny/2); radius=15;
for i=1:Nx, for j=1:Ny
    if (i-cx)^2+(j-cy)^2<=radius^2, mu_true(i,j)=9000; eta_true(i,j)=1.8; end
end, end

% FDTD -> truong SACH
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

% DFT @100Hz
L=nSteps; matran=dftmtx(L); fax=(0:L-1)/(L*dt); [~,idxf]=min(abs(fax-frq));
dft=@(v) matran(idxf,:)*v(:);

border=3; valid=false(Nx,Ny); valid(border+1:Nx-border,border+1:Ny-border)=true;

methods={'Hampel','Median(5)','LMS','Hampel->LMS'};
nM=numel(methods);
ampbias=zeros(nM,1); phaseerr=zeros(nM,1); wrmse=zeros(nM,1); repl=zeros(nM,1);
cnt=0;
ab=zeros(nM,1); pe=zeros(nM,1); wr=zeros(nM,1); rp=zeros(nM,1);

for i=1:Nx, for j=1:Ny
    if ~valid(i,j), continue; end
    cnt=cnt+1;
    s=squeeze(vz_time(:,i,j));                 % tin hieu SACH
    zref=zeros(nSteps,1);                        % khong co nhieu
    % cac bo loc
    h =hampel(s,k_hampel,threshold_hampel);
    m =medfilt1(s,5);
    l =LMS_filter_wt(s,zref,[],cfg_lms);  l=l(:);
    hl=LMS_filter_wt(h,zref,[],cfg_lms);  hl=hl(:);
    outs={h,m,l,hl};
    Vc=dft(s);
    for mm=1:nM
        o=outs{mm}; Vf=dft(o);
        ab(mm)=ab(mm)+abs( (abs(Vf)-abs(Vc))/(abs(Vc)+eps) );
        pe(mm)=pe(mm)+abs(angle(Vf/ (Vc+eps)))*180/pi;
        wr(mm)=wr(mm)+norm(o-s)/(norm(s)+eps);
        rp(mm)=rp(mm)+ sum(abs(o-s)>1e-12)/nSteps;   % ti le mau bi thay doi
    end
end, end

ab=ab/cnt*100; pe=pe/cnt; wr=wr/cnt*100; rp=rp/cnt*100;

fprintf('\n===== MEO TIN HIEU SACH (trung binh tren pixel hop le) =====\n');
fprintf('%-14s %12s %12s %12s %14s\n','Method','AmpBias(%)','PhaseErr(deg)','WaveRMSE(%)','ReplRate(%)');
for mm=1:nM
    fprintf('%-14s %12.3f %12.4f %12.3f %14.3f\n', methods{mm}, ab(mm), pe(mm), wr(mm), rp(mm));
end
fprintf('\n(AmpBias/PhaseErr @100Hz; WaveRMSE = ||loc-sach||/||sach||; ReplRate = %% mau bi thay doi)\n');
