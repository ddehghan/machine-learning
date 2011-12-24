function [J, grad] = costFunction(theta, X, y)
##COSTFUNCTION Compute cost and gradient for logistic regression
##   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
##   parameter for logistic regression and the gradient of the cost
##   w.r.t. to the parameters.

  m = length (y);

  ## h_theta(x)
  ht = sigmoid (X*theta); 

  J = - (y'*log (ht) + (1 - y)'*log (1 - ht))/m;
  grad = X'*(ht - y)/m;

endfunction
