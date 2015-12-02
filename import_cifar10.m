run('vlfeat-0.9.20/toolbox/vl_setup');

load('subset_CIFAR10/small_data_batch_1.mat');
data_1 = single(data);
label_1 = labels;
load('subset_CIFAR10/small_data_batch_2.mat');
data_2 = single(data);
label_2 = labels;
load('subset_CIFAR10/small_data_batch_3.mat');
data_3 = single(data);
label_3 = labels;
load('subset_CIFAR10/small_data_batch_4.mat');
data_4 = single(data);
label_4 = labels;
load('subset_CIFAR10/small_data_batch_5.mat');
data_5 = single(data);
label_5 = labels;

data = vertcat(data_1, data_2, data_3, data_4, data_5);
labels = vertcat(label_1, label_2, label_3, label_4, label_5);