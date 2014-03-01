clear;
load banana;
x = train.data;
y = train.labels;

sigma = 0.5;
[lambda, alpha, elbow] = svmPath(x, y, @kernel, sigma);
for i = 1:size(alpha, 1)
    plot(lambda.^(-1), alpha(i, :));
    hold all;
end
