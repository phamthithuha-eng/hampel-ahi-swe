function [out, w_final] = LMS_filter_wt(measured_signal, nhieu, w_init, cfg)
% LMS_FILTER_WT  Bo loc ANC (giu NGUYEN quy tac cap nhat goc) + TRUYEN TRONG SO.
%
%   [out, w_final] = LMS_filter_wt(measured_signal, nhieu, w_init, cfg)
%
%   measured_signal : chuoi tin hieu DO tai 1 diem (i,j)
%   nhieu           : chuoi THAM CHIEU nhieu tai cung diem (cau hinh oracle)
%   w_init          : trong so khoi tao (truyen tu cot truoc). [] = khoi tao 0.
%   cfg             : .order (mac dinh 2), .mu (mac dinh 0.2)
%
%   out      : chuoi da khu nhieu
%   w_final  : trong so hoi tu (de truyen sang cot ke tiep)
%
%   GIU NGUYEN cong thuc cap nhat cua ban goc (LMS_filter.m):
%       weight = weight + buffer * mu * e / norm(buffer)     (chuan hoa BAC 1)
%   vi tham chieu (nhieu) va tin hieu khac thang nhau rat lon -> KHONG dung ||x||^2.
%   CHI bo sung: (1) dung w_init, (2) tra ve w_final, (3) lap du chuoi (duoi = tin hieu do).

    if nargin < 4, cfg = struct(); end
    if ~isfield(cfg,'order'), cfg.order = 2;   end
    if ~isfield(cfg,'mu'),    cfg.mu    = 0.2; end
    order = cfg.order;  muy = cfg.mu;

    meas = measured_signal(:);                          % n x 1
    ref  = nhieu(:) + 0.00002*randn;                    % GIU NGUYEN ban goc: cong 1 so vo huong nho
                                                        % (KHONG bom nhieu tung mau -> khong pha ANC)
    n    = numel(meas);
    out  = meas;                                        % mac dinh = tin hieu do (duoi KHONG bi 0)

    % ---- khoi tao: TRUYEN trong so tu cot truoc neu co ----
    if isempty(w_init) || numel(w_init) ~= order
        weight = zeros(order,1);
    else
        weight = w_init(:);                             % <== truyen trong so khong gian
    end

    % ---- vong lap ANC (GIU NGUYEN quy tac goc: chuan hoa bac 1) ----
    for k = 1:n-order
        buffer = ref(k:k+order-1);                      % order x 1 (cung chieu index nhu ban goc)
        out(k) = meas(k) - buffer.'*weight;             % dau ra
        weight = weight + muy*out(k)*buffer/norm(buffer); % cap nhat (/ ||x|| bac 1)
    end

    w_final = weight;                                   % <== tra ve de truyen sang cot ke tiep
end
