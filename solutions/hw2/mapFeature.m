function out = mapFeature(X1, X2)
  ## MAPFEATURE Feature mapping function to polynomial features
  ##
  ##   MAPFEATURE(X1, X2) maps the two input features
  ##   to quadratic features used in the regularization exercise.
  ##
  ##   Returns a new feature array with more features, comprising of 
  ##   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
  ##
  ##   Inputs X1, X2 must be the same size
  ##

  degree = 6;

  ## Compute the powers with an upper-triangular matrix trick
  [i, j] = find (triu (ones (degree+1, degree+1)));
  i--; j--;
  [j, i] = deal (i', j');

  out = bsxfun (@power, X1, i - j).*bsxfun (@power, X2, j);
endfunction
