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

reg = 0
n = length(theta)
for i=1:m
  J = J + (y(i) * log(sigmoid((X*theta)(i)))) + (1-y(i))*log(1-(sigmoid((X*theta)(i))));
endfor
for i=2:n
  reg= reg + (theta(i)^2);
endfor
reg = (reg*lambda)/(2*m);
J = (-J/m) + reg;

for i=1:size(theta,1)
  for j=1:size(theta,2)
    for k=1:m
      grad(i,j) = grad(i,j) + (sigmoid((X*theta)(k)) - y(k))*X(k,i);
    endfor
    grad(i,j) = grad(i,j)/m;
  endfor
endfor

for i =2:size(theta,1)
  grad(i) = grad(i) + ((lambda*theta(i))/m);
endfor



% =============================================================

end
