function K = calcK(x, z, type, param)
%CALCK Compute the kernel function value of x and z
%   K = CALCK(X, type) compute the kernel function value of x and z.
%   type is the type number of kernel functions.
%   param is parameter for some kind of kernel functions.

if nargin < 3
    type = 0;
end
if nargin < 4
    param = 0;
end

switch type
    case 0
        % none kernel function been used
        K = x * z';
    case 1
        % polynomial kernel function, param is for p
        p = param;
        K = (x * z' + 1)^p;
    case 2
        % Gaussian kernel function, param is for sigma
        sigma = param;
        N = size(x, 1);
        M = size(z, 1);
        K = zeros(N, M);
        for i=1:N
            for j=1:M
                d = x(i, :) - z(j, :);
                K(i, j) = exp(-(d * d') / (2*sigma^2));
            end
        end
end

end

