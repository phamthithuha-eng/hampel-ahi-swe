clc; clear all; close all;

% ==================== THAM SỐ MÔ PHỎNG ====================
Nx = 80; Ny = 80;          % Kích thước lưới (giảm để chạy nhanh)
dx = 1e-3; dy = 1e-3;      % 1 mm
dt = 8e-5;                 % 0.08 ms
nSteps = 4000;             % Số bước thời gian
frq = 100;                 % Tần số kích thích 100 Hz
w = 2*pi*frq;
rho = 1000;                % kg/m^3
A = 0.005;                 % Biên độ nguồn

% ==================== MÔI TRƯỜNG: NỀN + KHỐI U ====================
mu_true = 6000 * ones(Nx, Ny);   % Độ cứng (Pa)
eta_true = 1.2 * ones(Nx, Ny);   % Độ nhớt (Pa·s)

% Chèn khối u hình tròn tại tâm (bán kính 15 điểm)
centerX = round(Nx/2); centerY = round(Ny/2);
radius = 15;
for i = 1:Nx
    for j = 1:Ny
        if (i - centerX)^2 + (j - centerY)^2 <= radius^2
            mu_true(i,j) = 9000;
            eta_true(i,j) = 1.8;
        end
    end
end

% Cố định giá trị biên cho hiển thị
mu_true(end,end) = 10000; mu_true(end,end-1) = 0;
eta_true(end,end) = 2;    eta_true(end,end-1) = 0;

% ==================== FDTD 2D (VISCOELASTIC) ====================
vz = zeros(Nx, Ny);
vz_time = zeros(nSteps, Nx, Ny);
temp = zeros(Nx, Ny);
sgm_zx = zeros(Nx, Ny);
sgm_zy = zeros(Nx, Ny);

for t = 1:nSteps
    % Nguồn tại góc trên trái
    vz(1,1) = A * cos(w*(t-1)*dt);
    vz(1,2) = A * cos(w*(t-1)*dt + dx/(2*pi));
    vz(2,1) = A * cos(w*(t-1)*dt + dx/(2*pi));
    vz(2,2) = A * cos(w*(t-1)*dt + dx/(2*pi));
    
    for i = 2:Nx-1
        for j = 2:Ny-1
            sgm_zx(i,j) = sgm_zx(i,j) + mu_true(i,j)*dt/dx*(vz(i+1,j)-vz(i,j)) ...
                        + eta_true(i,j)/dx*(vz(i+1,j)-vz(i,j)) ...
                        - eta_true(i,j)/dx*(temp(i+1,j)-temp(i,j));
            sgm_zy(i,j) = sgm_zy(i,j) + mu_true(i,j)*dt/dy*(vz(i,j+1)-vz(i,j)) ...
                        + eta_true(i,j)/dy*(vz(i,j+1)-vz(i,j)) ...
                        - eta_true(i,j)/dy*(temp(i,j+1)-temp(i,j));
            temp(i,j) = vz(i,j);
            vz(i,j) = vz(i,j) + dt/(rho*dx)*(sgm_zx(i,j)-sgm_zx(i-1,j)) ...
                               + dt/(rho*dy)*(sgm_zy(i,j)-sgm_zy(i,j-1));
        end
    end
    vz_time(t,:,:) = vz;
end

% ==================== THÊM NHIỄU GAUSSIAN + NHIỄU XUNG ====================
noise_gauss = 12e-7 * randn(nSteps, Nx, Ny);   % Nhiễu Gaussian nhẹ
noise_impulse = zeros(nSteps, Nx, Ny);
p_impulse = 0.002;        % Xác suất xuất hiện nhiễu xung (0.2%)
impulse_amplitude = 0.001;

for i = 1:Nx
    for j = 1:Ny
        rand_vals = rand(nSteps,1);
        idx = rand_vals < p_impulse;
        noise_impulse(idx,i,j) = impulse_amplitude * (2*randi([0,1], sum(idx),1)-1);
    end
end

noise_total = noise_gauss + noise_impulse;
vz_time_noisy = vz_time + noise_total;

% ==================== LỌC HAMPEL (TRƯỚC) ====================
k_hampel = 5;          % cửa sổ ±5 mẫu
threshold_hampel = 5;  % ngưỡng 5 sigma
vz_time_hampel_first = zeros(nSteps, Nx, Ny);
for i = 1:Nx
    for j = 1:Ny
        vz_time_hampel_first(:,i,j) = hampel(squeeze(vz_time_noisy(:,i,j)), k_hampel, threshold_hampel);
    end
end

% ==================== LỌC LMS (SAU HAMPEL) ====================
vz_time_lms_after = zeros(nSteps, Nx, Ny);
for i = 1:Nx
    for j = 1:Ny
        meas = squeeze(vz_time_hampel_first(:,i,j));   % tín hiệu đã qua Hampel
        nse  = squeeze(noise_total(:,i,j));            % tham chiếu nhiễu gốc
        if j == 1
            [desired, ~] = LMS_filter(meas, nse);
        else
            [desired, ~] = LMS_filter_2024(meas, nse, wlms_old);
        end
        vz_time_lms_after(:,i,j) = desired;
        if j == 1
            wlms_old = [];
        end
    end
end

% ==================== AHI (TẦN SỐ KÍCH THÍCH) ====================
L = nSteps;
matran = dftmtx(L);
freq_axis = (0:L-1)/(L*dt);
[~, idx_f] = min(abs(freq_axis - frq));

% Hàm AHI
function [mu_est, eta_est] = computeAHI(v_field, idx_f, matran, rho, w, dx)
    [~, Nx, Ny] = size(v_field);
    Vw = zeros(Nx, Ny);
    for i = 1:Nx
        for j = 1:Ny
            Vw(i,j) = matran(idx_f,:) * squeeze(v_field(:,i,j));
        end
    end
    Lap = 4*del2(Vw) / (dx^2);
    mu_complex = (-rho * w^2 * Vw) ./ Lap;
    mu_est = real(mu_complex);
    eta_est = imag(mu_complex) / w;
    % Gán biên cho hiển thị
    mu_est(end,end) = 10000; mu_est(end,end-1) = 0;
    eta_est(end,end) = 2;    eta_est(end,end-1) = 0;
end

% Chỉ LMS (không lọc Hampel) – giữ lại để so sánh
[mu_lms_only, eta_lms_only] = computeAHI(vz_time_noisy, idx_f, matran, rho, w, dx);

% Hampel → LMS
[mu_hampel_then_lms, eta_hampel_then_lms] = computeAHI(vz_time_lms_after, idx_f, matran, rho, w, dx);

% ==================== TÍNH SAI SỐ (VÙNG KHỐI U) ====================
r = radius;
row_range = max(1,centerX-r):min(Nx,centerX+r);
col_range = max(1,centerY-r):min(Ny,centerY+r);
mu_true_tumor = mu_true(row_range, col_range);
eta_true_tumor = eta_true(row_range, col_range);
mu_lms_tumor = mu_lms_only(row_range, col_range);
mu_hampel_tumor = mu_hampel_then_lms(row_range, col_range);
eta_lms_tumor = eta_lms_only(row_range, col_range);
eta_hampel_tumor = eta_hampel_then_lms(row_range, col_range);

err_lms_mu = mean(abs(mu_lms_tumor(:) - mu_true_tumor(:)) ./ mu_true_tumor(:));
err_lms_eta = mean(abs(eta_lms_tumor(:) - eta_true_tumor(:)) ./ eta_true_tumor(:));
err_hampel_mu = mean(abs(mu_hampel_tumor(:) - mu_true_tumor(:)) ./ mu_true_tumor(:));
err_hampel_eta = mean(abs(eta_hampel_tumor(:) - eta_true_tumor(:)) ./ eta_true_tumor(:));

fprintf('\n========== SAI SỐ TRÊN VÙNG KHỐI U ==========\n');
fprintf('Chỉ LMS:         μ: %.2f%%,   η: %.2f%%\n', err_lms_mu*100, err_lms_eta*100);
fprintf('Hampel → LMS:    μ: %.2f%%,   η: %.2f%%\n', err_hampel_mu*100, err_hampel_eta*100);

% ==================== VẼ HÌNH SO SÁNH ====================
% Hình 1: Ảnh đàn hồi lý tưởng
figure('Name','Ground Truth');
subplot(1,2,1); imagesc(mu_true); title('Độ cứng lý tưởng'); colorbar; axis equal;
subplot(1,2,2); imagesc(eta_true); title('Độ nhớt lý tưởng'); colorbar; axis equal;

% Hình 2: So sánh chỉ LMS vs Hampel→LMS
figure('Name','So sánh LMS vs Hampel-LMS');
subplot(2,2,1); imagesc(mu_lms_only); title('Độ cứng - Chỉ LMS'); colorbar; axis equal;
subplot(2,2,2); imagesc(mu_hampel_then_lms); title('Độ cứng - Hampel→LMS'); colorbar; axis equal;
subplot(2,2,3); imagesc(eta_lms_only); title('Độ nhớt - Chỉ LMS'); colorbar; axis equal;
subplot(2,2,4); imagesc(eta_hampel_then_lms); title('Độ nhớt - Hampel→LMS'); colorbar; axis equal;
sgtitle('So sánh thứ tự lọc: Hampel trước LMS');

% Hình 3: Dạng sóng thời gian tại một điểm trong khối u
ptX = centerX; ptY = centerY;
figure('Name','Dạng sóng thời gian');
plot(vz_time(:,ptX,ptY), 'k', 'LineWidth',1); hold on;
plot(vz_time_noisy(:,ptX,ptY), 'b');
plot(vz_time_hampel_first(:,ptX,ptY), 'm');
plot(vz_time_lms_after(:,ptX,ptY), 'g');
legend('Lý tưởng','Nhiễu + Xung','Sau Hampel','Hampel→LMS');
xlabel('Bước thời gian'); ylabel('Vận tốc (m/s)');
title(sprintf('Tín hiệu tại tâm khối u (%d,%d)', ptX, ptY));
grid on;

% Hình 4: Zoom vào đoạn có nhiễu xung
figure('Name','Zoom nhiễu xung');
t_range = 1000:1100;
plot(t_range, vz_time(t_range,ptX,ptY), 'k-', 'LineWidth',1.5); hold on;
plot(t_range, vz_time_noisy(t_range,ptX,ptY), 'b-');
plot(t_range, vz_time_hampel_first(t_range,ptX,ptY), 'm-');
plot(t_range, vz_time_lms_after(t_range,ptX,ptY), 'g-', 'LineWidth',1.2);
legend('Lý tưởng','Nhiễu + Xung','Sau Hampel','Hampel→LMS');
xlabel('Bước thời gian'); ylabel('Vận tốc (m/s)');
title('Loại bỏ nhiễu xung bằng Hampel trước LMS');
grid on;

% ==================== CÁC CHỈ SỐ ĐÁNH GIÁ ẢNH ====================
mask = true(Nx, Ny);
mask(end, end) = false;
mask(end, end-1) = false;

function [mse, psnr, nrmse, corr_coef] = calc_image_metrics(img_true, img_est, mask)
    img_true = double(img_true(mask));
    img_est  = double(img_est(mask));
    mse = mean((img_true - img_est).^2);
    max_val = max(img_true);
    min_val = min(img_true);
    psnr = 10 * log10(max_val^2 / mse);
    nrmse = sqrt(mse) / (max_val - min_val);
    corr_coef = corr(img_true, img_est);
end

% Tính chỉ số cho μ, η
[mu_lms_mse, mu_lms_psnr, mu_lms_nrmse, mu_lms_corr] = ...
    calc_image_metrics(mu_true, mu_lms_only, mask);
[mu_hampel_mse, mu_hampel_psnr, mu_hampel_nrmse, mu_hampel_corr] = ...
    calc_image_metrics(mu_true, mu_hampel_then_lms, mask);
[eta_lms_mse, eta_lms_psnr, eta_lms_nrmse, eta_lms_corr] = ...
    calc_image_metrics(eta_true, eta_lms_only, mask);
[eta_hampel_mse, eta_hampel_psnr, eta_hampel_nrmse, eta_hampel_corr] = ...
    calc_image_metrics(eta_true, eta_hampel_then_lms, mask);

% SSIM
try
    [mu_lms_ssim, ~] = ssim(mu_lms_only, mu_true, 'DynamicRange', max(mu_true(:))-min(mu_true(:)));
    [mu_hampel_ssim, ~] = ssim(mu_hampel_then_lms, mu_true, 'DynamicRange', max(mu_true(:))-min(mu_true(:)));
    [eta_lms_ssim, ~] = ssim(eta_lms_only, eta_true, 'DynamicRange', max(eta_true(:))-min(eta_true(:)));
    [eta_hampel_ssim, ~] = ssim(eta_hampel_then_lms, eta_true, 'DynamicRange', max(eta_true(:))-min(eta_true(:)));
    ssim_available = true;
catch
    ssim_available = false;
    warning('Không thể tính SSIM (có thể thiếu Image Processing Toolbox)');
end

fprintf('\n========== ĐÁNH GIÁ CHẤT LƯỢNG ẢNH ==========\n');
fprintf('--- Độ cứng μ ---\n');
fprintf('Chỉ LMS:      MSE=%.3e, PSNR=%.2f dB, NRMSE=%.4f, Corr=%.4f', mu_lms_mse, mu_lms_psnr, mu_lms_nrmse, mu_lms_corr);
if ssim_available, fprintf(', SSIM=%.4f\n', mu_lms_ssim); else fprintf('\n'); end
fprintf('Hampel→LMS:   MSE=%.3e, PSNR=%.2f dB, NRMSE=%.4f, Corr=%.4f', mu_hampel_mse, mu_hampel_psnr, mu_hampel_nrmse, mu_hampel_corr);
if ssim_available, fprintf(', SSIM=%.4f\n', mu_hampel_ssim); else fprintf('\n'); end

fprintf('\n--- Độ nhớt η ---\n');
fprintf('Chỉ LMS:      MSE=%.3e, PSNR=%.2f dB, NRMSE=%.4f, Corr=%.4f', eta_lms_mse, eta_lms_psnr, eta_lms_nrmse, eta_lms_corr);
if ssim_available, fprintf(', SSIM=%.4f\n', eta_lms_ssim); else fprintf('\n'); end
fprintf('Hampel→LMS:   MSE=%.3e, PSNR=%.2f dB, NRMSE=%.4f, Corr=%.4f', eta_hampel_mse, eta_hampel_psnr, eta_hampel_nrmse, eta_hampel_corr);
if ssim_available, fprintf(', SSIM=%.4f\n', eta_hampel_ssim); else fprintf('\n'); end

% Contrast-to-Noise Ratio (CNR)
tumor_mask = false(Nx, Ny);
for i = 1:Nx
    for j = 1:Ny
        if (i - centerX)^2 + (j - centerY)^2 <= radius^2
            tumor_mask(i,j) = true;
        end
    end
end
background_mask = mask & ~tumor_mask;

function cnr = compute_CNR(img, tumor_mask, background_mask)
    img_tumor = img(tumor_mask);
    img_bg = img(background_mask);
    mean_tumor = mean(img_tumor);
    mean_bg = mean(img_bg);
    std_bg = std(img_bg);
    cnr = abs(mean_tumor - mean_bg) / std_bg;
end

cnr_mu_true = compute_CNR(mu_true, tumor_mask, background_mask);
cnr_mu_lms = compute_CNR(mu_lms_only, tumor_mask, background_mask);
cnr_mu_hampel = compute_CNR(mu_hampel_then_lms, tumor_mask, background_mask);
cnr_eta_true = compute_CNR(eta_true, tumor_mask, background_mask);
cnr_eta_lms = compute_CNR(eta_lms_only, tumor_mask, background_mask);
cnr_eta_hampel = compute_CNR(eta_hampel_then_lms, tumor_mask, background_mask);

fprintf('\n========== TỶ SỐ TƯƠNG PHẢN TRÊN NHIỄU (CNR) ==========\n');
fprintf('μ - Ground truth: %.2f\n', cnr_mu_true);
fprintf('μ - Chỉ LMS:       %.2f\n', cnr_mu_lms);
fprintf('μ - Hampel→LMS:    %.2f\n', cnr_mu_hampel);
fprintf('η - Ground truth: %.2f\n', cnr_eta_true);
fprintf('η - Chỉ LMS:       %.2f\n', cnr_eta_lms);
fprintf('η - Hampel→LMS:    %.2f\n', cnr_eta_hampel);

% ==================== 1. ĐƯỜNG CONG HỘI TỤ MSE (LEARNING CURVE) ====================
% Tính MSE theo thời gian cho vùng khối u
tumor_indices = find(tumor_mask);
n_tumor_pixels = length(tumor_indices);

% Khởi tạo mảng lưu MSE theo thời gian
mse_lms_time = zeros(nSteps, 1);      % MSE của LMS (không lọc Hampel)
mse_hampel_lms_time = zeros(nSteps, 1); % MSE của Hampel → LMS

for t = 1:nSteps
    % Lấy tín hiệu tại thời điểm t cho tất cả pixel
    ideal_signal = squeeze(vz_time(t, :, :));
    lms_signal = squeeze(vz_time_noisy(t, :, :));      % Chỉ LMS (không lọc)
    hampel_lms_signal = squeeze(vz_time_lms_after(t, :, :)); % Hampel → LMS
    
    % Chỉ lấy các pixel trong vùng u
    ideal_tumor = ideal_signal(tumor_mask);
    lms_tumor = lms_signal(tumor_mask);
    hampel_lms_tumor = hampel_lms_signal(tumor_mask);
    
    % Tính MSE tại thời điểm t
    mse_lms_time(t) = mean((ideal_tumor - lms_tumor).^2);
    mse_hampel_lms_time(t) = mean((ideal_tumor - hampel_lms_tumor).^2);
end

% Vẽ Learning Curve
figure('Name', 'Đường cong hội tụ MSE');
set(gcf, 'Position', [100, 100, 1200, 500]);

% Subplot 1: MSE theo thời gian (thang log)
subplot(1, 2, 1);
semilogy(1:nSteps, mse_lms_time, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Chỉ LMS'); hold on;
semilogy(1:nSteps, mse_hampel_lms_time, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Hampel → LMS');

% Thêm đường trung bình động
window_size = 100;
mse_lms_smooth = movmean(mse_lms_time, window_size);
mse_hampel_lms_smooth = movmean(mse_hampel_lms_time, window_size);
semilogy(1:nSteps, mse_lms_smooth, 'b--', 'LineWidth', 2.5, 'DisplayName', 'LMS (TB động)');
semilogy(1:nSteps, mse_hampel_lms_smooth, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Hampel→LMS (TB động)');

xlabel('Bước thời gian');
ylabel('Sai số bình phương trung bình (MSE)');
title('MSE theo thời gian (thang log)');
legend('Location', 'northeast');
grid on;
set(gca, 'FontSize', 11);
xlim([1, nSteps]);

% Đánh dấu vùng ổn định
steady_start = round(nSteps * 0.7);
patch([steady_start, nSteps, nSteps, steady_start], ...
      [10^-12, 10^-12, 10^-2, 10^-2], 'g', 'FaceAlpha', 0.05, 'EdgeColor', 'none');
text(steady_start + nSteps*0.05, 10^-6, 'Vùng ổn định', 'Color', 'g', 'FontWeight', 'bold');

% Subplot 2: Cumulative MSE
cum_mse_lms = cumsum(mse_lms_time) ./ (1:nSteps)';
cum_mse_hampel_lms = cumsum(mse_hampel_lms_time) ./ (1:nSteps)';
subplot(1, 2, 2);
semilogy(1:nSteps, cum_mse_lms, 'b-', 'LineWidth', 2, 'DisplayName', 'Chỉ LMS'); hold on;
semilogy(1:nSteps, cum_mse_hampel_lms, 'r-', 'LineWidth', 2, 'DisplayName', 'Hampel → LMS');
xlabel('Bước thời gian');
ylabel('MSE tích lũy');
title('MSE tích lũy (tốc độ hội tụ)');
legend('Location', 'northeast');
grid on;
set(gca, 'FontSize', 11);
xlim([1, nSteps]);

sgtitle('Đường cong học: So sánh hội tụ MSE giữa LMS và Hampel→LMS');

% ==================== 2. PHÂN BỐ NHIỄU TRƯỚC VÀ SAU HAMPEL ====================
% Chọn một điểm trong khối u để phân tích histograms
ptX = centerX; ptY = centerY;

% Trích xuất tín hiệu tại điểm đó
signal_ideal = squeeze(vz_time(:, ptX, ptY));
signal_noisy = squeeze(vz_time_noisy(:, ptX, ptY));
signal_hampel = squeeze(vz_time_hampel_first(:, ptX, ptY));

% Tính noise trước và sau lọc
noise_before = signal_noisy - signal_ideal;
noise_after = signal_hampel - signal_ideal;

% Vẽ histograms
figure('Name', 'Phân bố nhiễu trước và sau Hampel');
set(gcf, 'Position', [150, 150, 1000, 400]);

subplot(1, 2, 1);
histogram(noise_before, 50, 'Normalization', 'pdf', 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
% Fit Gaussian để so sánh
x_vals = linspace(min(noise_before), max(noise_before), 100);
mu_noise = mean(noise_before);
sigma_noise = std(noise_before);
gauss_fit = (1/(sigma_noise*sqrt(2*pi))) * exp(-(x_vals - mu_noise).^2/(2*sigma_noise^2));
plot(x_vals, gauss_fit, 'r-', 'LineWidth', 2);
xlabel('Biên độ nhiễu');
ylabel('Mật độ xác suất');
title(sprintf('Trước Hampel (Đuôi nặng)\nTrung bình=%.3e, Độ lệch=%.3e', mu_noise, sigma_noise));
legend('Histogram nhiễu', 'Phân bố Gaussian', 'Location', 'northwest');
grid on;

subplot(1, 2, 2);
histogram(noise_after, 50, 'Normalization', 'pdf', 'FaceColor', 'g', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
% Fit Gaussian cho noise sau lọc
x_vals_after = linspace(min(noise_after), max(noise_after), 100);
mu_noise_after = mean(noise_after);
sigma_noise_after = std(noise_after);
gauss_fit_after = (1/(sigma_noise_after*sqrt(2*pi))) * exp(-(x_vals_after - mu_noise_after).^2/(2*sigma_noise_after^2));
plot(x_vals_after, gauss_fit_after, 'r-', 'LineWidth', 2);
xlabel('Biên độ nhiễu');
ylabel('Mật độ xác suất');
title(sprintf('Sau Hampel (Gaussian cắt ngắn)\nTrung bình=%.3e, Độ lệch=%.3e', mu_noise_after, sigma_noise_after));
legend('Histogram nhiễu', 'Phân bố Gaussian', 'Location', 'northwest');
grid on;
sgtitle('So sánh phân bố nhiễu: Hiệu quả của bộ lọc Hampel');

% Thống kê về nhiễu xung
impulse_indices = find(abs(noise_before) > 3*std(noise_after));
fprintf('\n========== THỐNG KÊ NHIỄU ==========\n');
fprintf('Số điểm nhiễu xung: %d / %d (%.2f%%)\n', ...
    length(impulse_indices), nSteps, length(impulse_indices)/nSteps*100);
fprintf('Trước Hampel: Trung bình=%.2e, Độ lệch=%.2e, Max=%.2e\n', ...
    mean(noise_before), std(noise_before), max(abs(noise_before)));
fprintf('Sau Hampel:  Trung bình=%.2e, Độ lệch=%.2e, Max=%.2e\n', ...
    mean(noise_after), std(noise_after), max(abs(noise_after)));
fprintf('Giảm nhiễu: %.2f%%\n', (1 - std(noise_after)/std(noise_before))*100);

% ==================== 3. MẶT CẮT 1D VÀ BẢO TOÀN BIÊN (EDGE PRESERVATION) ====================
% Tạo mặt cắt 1D đi qua tâm khối u theo chiều ngang (dòng centerX)
row_cut = centerX;
mu_true_cut = mu_true(row_cut, :);
mu_lms_cut = mu_lms_only(row_cut, :);
mu_hampel_cut = mu_hampel_then_lms(row_cut, :);

% Áp dụng bộ lọc Median truyền thống để so sánh
mu_median_cut = medfilt1(mu_lms_cut, 5);  % Cửa sổ 5 điểm

% Tính gradient (đạo hàm bậc nhất)
grad_true = gradient(mu_true_cut);
grad_lms = gradient(mu_lms_cut);
grad_hampel = gradient(mu_hampel_cut);
grad_median = gradient(mu_median_cut);

% Vẽ 1D Gradient Profile
figure('Name', 'Mặt cắt 1D và bảo toàn biên');
set(gcf, 'Position', [200, 200, 1000, 600]);

subplot(2, 1, 1);
plot(1:Ny, mu_true_cut, 'k-', 'LineWidth', 2); hold on;
plot(1:Ny, mu_lms_cut, 'b-', 'LineWidth', 1.5);
plot(1:Ny, mu_hampel_cut, 'r-', 'LineWidth', 1.5);
plot(1:Ny, mu_median_cut, 'g--', 'LineWidth', 1.5);
xlabel('Vị trí pixel (ngang qua khối u)');
ylabel('Độ cứng μ (Pa)');
title('Mặt cắt 1D qua tâm khối u (cắt ngang)');
legend('Ground Truth', 'Chỉ LMS', 'Hampel → LMS', 'Bộ lọc Median (5-pt)', 'Location', 'best');
grid on;
xlim([1, Ny]);
ylim([min(mu_true_cut)-500, max(mu_true_cut)+500]);

% Đánh dấu vùng khối u
y_limits = ylim;
hold on;
patch([col_range(1), col_range(end), col_range(end), col_range(1)], ...
      [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
text(mean(col_range), y_limits(2)*0.95, 'Vùng khối u', 'Color', 'r', 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');

subplot(2, 1, 2);
plot(1:Ny, grad_true, 'k-', 'LineWidth', 2); hold on;
plot(1:Ny, grad_lms, 'b-', 'LineWidth', 1.5);
plot(1:Ny, grad_hampel, 'r-', 'LineWidth', 1.5);
plot(1:Ny, grad_median, 'g--', 'LineWidth', 1.5);
xlabel('Vị trí pixel (ngang qua khối u)');
ylabel('Gradient (dμ/dx)');
title('Đạo hàm bậc nhất (phát hiện biên)');
legend('Ground Truth', 'Chỉ LMS', 'Hampel → LMS', 'Bộ lọc Median (5-pt)', 'Location', 'best');
grid on;
xlim([1, Ny]);

% Đánh dấu vùng khối u
y_limits = ylim;
hold on;
patch([col_range(1), col_range(end), col_range(end), col_range(1)], ...
      [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
text(mean(col_range), y_limits(2)*0.9, 'Vùng khối u', 'Color', 'r', 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'center');
sgtitle('Phân tích bảo toàn biên: So sánh Hampel với bộ lọc Median');

% Tính chỉ số bảo toàn biên (Edge Preservation Index - EPI)
function epi = compute_EPI(grad_est, grad_true)
    epi = norm(grad_est)^2 / norm(grad_true)^2;
end

epi_lms = compute_EPI(grad_lms, grad_true);
epi_hampel = compute_EPI(grad_hampel, grad_true);
epi_median = compute_EPI(grad_median, grad_true);

fprintf('\n========== CHỈ SỐ BẢO TOÀN BIÊN (EPI) ==========\n');
fprintf('Chỉ LMS:        EPI = %.4f\n', epi_lms);
fprintf('Hampel → LMS:   EPI = %.4f\n', epi_hampel);
fprintf('Bộ lọc Median:  EPI = %.4f\n', epi_median);
fprintf('(EPI = 1.0 nghĩa là bảo toàn biên hoàn hảo)\n');

% Phân tích độ dốc tại biên khối u
edge_left = col_range(1) + 2;
edge_right = col_range(end) - 2;

slope_true_left = (mu_true_cut(edge_left+1) - mu_true_cut(edge_left-1)) / 2;
slope_lms_left = (mu_lms_cut(edge_left+1) - mu_lms_cut(edge_left-1)) / 2;
slope_hampel_left = (mu_hampel_cut(edge_left+1) - mu_hampel_cut(edge_left-1)) / 2;
slope_median_left = (mu_median_cut(edge_left+1) - mu_median_cut(edge_left-1)) / 2;

slope_true_right = (mu_true_cut(edge_right+1) - mu_true_cut(edge_right-1)) / 2;
slope_lms_right = (mu_lms_cut(edge_right+1) - mu_lms_cut(edge_right-1)) / 2;
slope_hampel_right = (mu_hampel_cut(edge_right+1) - mu_hampel_cut(edge_right-1)) / 2;
slope_median_right = (mu_median_cut(edge_right+1) - mu_median_cut(edge_right-1)) / 2;

fprintf('\n========== PHÂN TÍCH ĐỘ DỐC TẠI BIÊN ==========\n');
fprintf('Biên trái:  Ground Truth=%.2f, LMS=%.2f, Hampel=%.2f, Median=%.2f\n', ...
    slope_true_left, slope_lms_left, slope_hampel_left, slope_median_left);
fprintf('Biên phải:  Ground Truth=%.2f, LMS=%.2f, Hampel=%.2f, Median=%.2f\n', ...
    slope_true_right, slope_lms_right, slope_hampel_right, slope_median_right);