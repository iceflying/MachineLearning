function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. It also works even z is a matrix or a vector.
%   In particular, if z is a vector or matrix, it return the gradient
%   for each element.

g = sigmoid(z) .* (1-sigmoid(z));

end
