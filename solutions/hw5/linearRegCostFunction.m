function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  ##LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
  ##regression with multiple variables
  ##   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
  ##   cost of using theta as the parameter for linear regression to fit the 
  ##   data points in X and y. Returns the cost in J and the gradient in grad

  m = length (y);
  ht = X*theta;
  J = (sumsq (ht - y) + lambda*sumsq (theta(2:end)))/(2*m);

  grad = (X'*(ht - y) + [0; lambda*theta(2:end)])/m;

endfunction
