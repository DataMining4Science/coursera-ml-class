function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % Vectorized hypothesis function (for all training data)
    _h = X * theta;

    % The delta of actual output of h(x) minus expected output (y)
    _delta = _h-y;

    % The partial derivative of the cost function. Note that the vectorized
    % implementation requires us to flip (h(x)-y)*x, since we do elementwise
    % multiplication of the whole matrix X by vector _delta in one shot.
    derivative = (1/m) * sum(X .* _delta);

    % Transpose theta and subtract the derivative with a step size determined
    % by the learning rate alpha.
    _theta = theta' - alpha * derivative;

    % Make _theta a m*1 matrix and assign it to the original reference of theta.
    theta = _theta';

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
