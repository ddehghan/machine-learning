function g = sigmoidGradient(z)
##SIGMOIDGRADIENT returns the gradient of the sigmoid function
##evaluated at z
##   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
##   evaluated at z. This should work regardless if z is a matrix or a
##   vector. In particular, if z is a vector or matrix, you should return
##   the gradient for each element.

  s = @(z) 1./(1 + exp(-z));
  g = s(z).*(1 - s(z));

endfunction
