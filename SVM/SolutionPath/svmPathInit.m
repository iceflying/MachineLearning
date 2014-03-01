function [lambda, alpha, Elbow] = init(y, Kt, gamma)
%INIT initialize the algorithm

EPS = 1e-10;
alpha = gamma*(y==1) + (1-gamma)*(y==-1);
gk = Kt * alpha;    % gk(i) == sum_j(alpha(j) * y(j) * K(i,j))

lambda = max(gk);
Elbow = find(lambda - gk < EPS);
