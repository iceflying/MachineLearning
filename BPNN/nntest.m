clear;

load('data.mat');

lambda = 1;
ita = 0.6;
num_iter = 100;
num_labels = 10;

% pred = bpnn(X, y, lambda, ita, num_iter, num_labels, Xtest);
% 
% fprintf('Training Set Accuracy: %f\n', mean(pred == ystand));

base_lambda = 0.1;
acc_lambda = zeros(20, 20);
for i=0:19
    for j=1:20
        pred = bpnn(X, y, i*base_lambda, ita, num_iter, num_labels, Xtest);
        acc_lambda(i+1, j) = mean(pred == ystand);
    end
end
subplot(1, 3, 1);
plot([0:19]*base_lambda, mean(acc_lambda, 2));

base_ita = 0.1;
acc_ita = zeros(20, 20);
for i=1:20
    for j=1:20
        pred = bpnn(X, y, lambda, i * base_ita, num_iter, num_labels, Xtest);
        acc_ita(i, j) = mean(pred == ystand);
    end
end
subplot(1, 3, 2);
plot([1:20]*base_ita, mean(acc_ita, 2));

base_num_iter = 25;
acc_iter = zeros(20, 20);
for i=1:20
    for j=1:20
        disp([i, j]);
        pred = bpnn(X, y, lambda, ita, i*base_num_iter, num_labels, Xtest);
        acc_iter(i, j) = mean(pred == ystand);
    end
end
subplot(1, 3, 3);
plot([1:20]*base_num_iter, mean(acc_iter, 2));

save -v7 solution.mat
