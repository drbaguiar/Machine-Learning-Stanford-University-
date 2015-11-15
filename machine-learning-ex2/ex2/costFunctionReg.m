function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1 : m
    temp = theta' * X(i,:)'
    J = J - y(i,:) * log(1/(1 + exp(-theta' * X(i,:)'))) - (1 - y(i,:)) * log(1 - 1/(1 + exp(-theta' * X(i,:)')))
end

J = J * (1 / m)

temp_J = 0
for i = 2:size(theta)
    temp_J = temp_J + theta(i).^2
end 
J = J + lambda/(2*m) * temp_J

for j = 1 : size(theta)
    for i = 1 : m
        grad(j) = grad(j) + ( 1 / (1 + exp(-theta' * X(i,:)')) - y(i,:)) * X(i,j)
    end
end
    
for j = 1: size(theta)
    grad(j) = (1/m) * grad(j)
end

for j = 2: size(theta)
    grad(j) = grad(j) + lambda/m * theta(j)
end
% =============================================================

end
