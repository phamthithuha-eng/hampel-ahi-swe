function [desired_signal, wlms] = LMS_filter_2024(measured_signal,nhieu,wlms_old)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
n = length(measured_signal);
N = n-1;
order = 2;
desired_signal = zeros(1,n);
ref = nhieu'+0.00002*randn;
weight=zeros(order,1);
muy=0.2;
wlms=wlms_old;
for k=1:N-order
   wlms=[wlms weight]; 
   buffer = ref(k:k+order-1);                                     %current 32 points of reference
   desired_signal(k) = measured_signal(k)-buffer*weight;          %dot product reference and coeffs
   weight=weight+(buffer.*muy*desired_signal(k)/norm(buffer))';   %update coeffs
end
end
