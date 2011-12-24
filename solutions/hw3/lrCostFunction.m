function [J, grad] = lrCostFunction(theta, X, y, lambda)
  ##LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
  ##regularization
  ##   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
  ##   theta as the parameter for regularized logistic regression and the
  ##   gradient of the cost w.r.t. to the parameters. 

  ## Initialize some useful values
  m = length(y); ## number of training examples

  ## h_theta(x)
  ht = sigmoid (X*theta); 

  J = -(y'*log (ht) + (1 - y)'*log (1 - ht))/m \
      + lambda*sumsq (theta(2:end))/(2*m);

  grad = (X'*(ht - y) + [0; lambda*theta(2:end,:)])/m ;

endfunction
