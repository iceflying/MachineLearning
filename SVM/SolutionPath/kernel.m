function K = kernel(x, y, sigma)
%KERNEL compute the Gaussian kernel matrix
%   K = kernel(x, y, sigma) compute the Gaussian kernel matirx K
%   x       N * p feature vector
%   y       M * p feature vector
%   sigma   kernel parameter
%   K(i, j) = exp(-(x(i)-y(j))^2 / 2*sigma)

normx = normsqr(x);
normy = normsqr(y);
[X1, Y1] = ndgrid(normx, normy);
K = exp(-(X1-2*x*y'+Y1)/(2*sigma^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = normsqr(x)
%NORMSQR auxiliary function for kernel function
%   x       N * p feature vector

y = sum(x.^2, 2);   % N * 1
