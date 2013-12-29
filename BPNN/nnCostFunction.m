function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

% Generate Y
transmap = eye(num_labels);
Y = transmap(y, :);                         % m * nl

% Compute feedforword neural network
a1 = [ones(1, size(X, 1)); X'];             % (n+1) * m
z2 = Theta1 * a1;                           % hs * m
a2 = [ones(1, size(z2, 2)); sigmoid(z2)];   % (hs+1) * m
z3 = Theta2 * a2;                           % nl * m
a3 = sigmoid(z3);                           % nl * m
hx = a3';                                   % m * nl

% Compute cost function
J = 0;
for k=1:num_labels
    J = J + (-Y(:, k)'*log(hx(:, k)) - (1-Y(:, k)')*log(1-hx(:, k)));
end
J = J / m;

% Regularized cost function
J = J + lambda / (2*m) * ...
    (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

% Compute backpropagation neural network 
delta3 = a3 - Y';                                           % nl * m
delta2 = Theta2(:, 2:end)'*delta3 .* sigmoidGradient(z2);   % hs * m

% Compute gradient
Theta1_grad = delta2 * a1' / m;                             % hs * (n+1)
Theta2_grad = delta3 * a2' / m;                             % nl * (hs+1)

% Regularized gradient
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ...
                        lambda / m * Theta1(:, 2:end);      % hs * (n+1)
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ...
                        lambda / m * Theta2(:, 2:end);      % nl * (hs+1)

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
