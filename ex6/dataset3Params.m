function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_try = [0.01;0.1;1;10;0.03;0.3;3;30];
sigma_try = [0.03;0.3;3;30;0.01;0.1;1;10];
error = zeros(8,1);
for i = 1:8
  model = svmTrain(X,y, C_try(i), @(x1, x2) gaussianKernel(x1, x2, sigma_try(i)));
  predictions = svmPredict(model, Xval);
  error(i) = mean(double(predictions ~=yval));
endfor

[x,ix] = min(error);
C = C_try(ix);
sigma = sigma_try(ix);

% =========================================================================

end
