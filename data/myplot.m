%% CLEAN WORKSPACE

clear; clc; close all;

markers = ['p', 'o','*','x', '+'];
colors = ['b', 'k','c','m', 'r'];


%% LOAD DATA



for kernel1 = 1:18
if kernel1 ~= 18
    kernel2=kernel1+1;
    kernel2_name = "Kernel"+num2str(kernel2);
else
    kernel2=0;
    kernel2_name = "Intel MKL";
end
data_tmp = zeros(30, 2);
array_size = [100:100:3000];
array_size = repmat(array_size, 2, 1);
data_ref_1_path     = "perf_"+num2str(kernel1)+".txt";
data_ref_2_path  = "perf_"+num2str(kernel2)+".txt";

kernel1_name = "Kernel"+num2str(kernel1);


data_ref_1 = load(data_ref_1_path);
data_ref_2 = load(data_ref_2_path);

min_len = min(length(data_ref_1),length(data_ref_2));
figure
plot(array_size(1, 1:length(data_ref_1))', data_ref_1, strcat('-', markers(1), colors(1)), 'LineWidth',2);
hold on
plot(array_size(2, 1:length(data_ref_2))', data_ref_2, strcat('-', markers(2), colors(2)), 'LineWidth',2);

if kernel1 <= 8
    legend(kernel1_name,kernel2_name,'Location','northeast', 'FontSize', 16)
else
    legend(kernel1_name,kernel2_name,'Location','southeast', 'FontSize', 16)
end
xlabel('Matrix Sizes (m=n=k)', 'FontSize', 16, 'FontWeight', 'bold')
ylabel('Performance (GFLOPS)', 'FontSize', 16, 'FontWeight', 'bold')
if kernel1 ~= 19 
    msg = "Comparison bewteen: Kernel " + num2str(kernel1) + " and Kernel " + num2str(kernel2);
else
    msg = "Comparison bewteen: Kernel " + num2str(kernel1) + " and Intel MKL";
end
if min_len==30 
    xlim([100,3000])
    xticks(100 : 100 : 3000);
else
    xlim([100,1000])
    xticks(100 : 100 : 1000);
end
ylim([0, 120])
a = get(gca,'XTickLabel');
set(gca,'XTickLabelRotation',45);
set(gca,'XTickLabel',a,'fontsize',12, 'FontWeight', 'bold')
title(sprintf(msg));
saveas(gca,kernel1_name+".png");
end




