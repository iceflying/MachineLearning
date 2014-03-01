clear;
load banana;
D = train.data;
yD = train.labels;
V = test.data;
yV = test.labels;

sigma = 0.5;
[lambda, alpha, elbow] = svmPath(D, yD, @kernel, sigma);
[Estar, lambdastar] = DGOP(lambda, alpha, D, yD, V, yV, @kernel, sigma);
disp(['Estar: ', num2str(Estar)])
disp(['lambdastar: ', num2str(lambdastar)]);
