% ============================================================
%  KHOI 6 (THAY THE) - LOC LMS SAU HAMPEL
%  Dan doan nay thay cho dong 90-106 cu trong Hample_LMS_RIMNI_Final.m
%  (bo dung LMS_filter.m / LMS_filter_2024.m cu; dung LMS_filter_wt.m)
% ============================================================
USE_TRANSFER = false;   % <== BUOC 1: de FALSE de TAI TAO ket qua tot cu (khoi tao 0 moi cot)
                        %     BUOC 2: doi thanh TRUE de BAT truyen trong so khong gian
cfg_lms = struct('order', 2, 'mu', 0.2);
vz_time_lms_after = zeros(nSteps, Nx, Ny);

for i = 1:Nx
    wprev = [];                          % dau moi hang: chua co trong so truoc
    for j = 1:Ny
        meas = squeeze(vz_time_hampel_first(:,i,j));   % tin hieu da qua Hampel
        nse  = squeeze(noise_total(:,i,j));            % tham chieu nhieu (oracle)
        if USE_TRANSFER
            [desired, wprev] = LMS_filter_wt(meas, nse, wprev, cfg_lms);  % truyen trong so
        else
            [desired, ~]     = LMS_filter_wt(meas, nse, [],    cfg_lms);  % khoi tao 0 moi cot
        end
        vz_time_lms_after(:,i,j) = desired;
    end
end
