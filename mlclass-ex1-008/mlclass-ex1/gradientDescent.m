function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %

    % This vector of ones is used for calculation of theta zero
    _ones = ones(m, 1);

    % Extract the features/input
    _x = X(:,2);

    % Our vectorized hypothesis function
    _h = X * theta;

    % This is the derivative of the MSE functions.
    % For theta zero, x is a vector of ones which keeps the values unchanged.
    % For theta one, x is the feature vetor.
    derivative = @(x) ((1/m) * sum((_h-y) .* x));

    % The heart of the gradient descent algorithm, where alpha is the step size
    % towards the global optimum (minimum) and the derivative function determines
    % the steepest slope, which is the direction we head towards.
    _theta_0 = theta(1) - alpha * derivative(_ones);
    _theta_1 = theta(2) - alpha * derivative(_x);

    theta = [_theta_0; _theta_1];

    % For debugging purposes
    %theta
    %fprintf('cost = %f\n', computeCost(X, y, theta));
    %fprintf('---------------\n');

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
