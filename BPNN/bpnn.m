function pred = bpnn(X, y, lambda, ita, num_iter, num_labels, Xtest)
%BPNN Implements the backpropagation neural network
%   pred = BPNN(X, y, lambda, num_iter, num_labels, Xtest) trains
%   backpropogation neural network using training dataset X and classifies
%   test dataset Xtest and return the prediction result.

% Setup the parameters
input_layer_size  = size(X, 2);
hidden_layer_size = 25;             % 25 hidden units

% Initializing thetas
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Initializing Params
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
% % options = optimset('MaxIter', num_iter);

% Minimize the cost function
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
% [nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

nn_params = initial_nn_params;
for i = 1:num_iter
    [J, grad] = costFunction(nn_params);
    nn_params = nn_params - ita * grad;
end

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Predict labels
pred = predict(Theta1, Theta2, Xtest);

end
