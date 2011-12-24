function [C, sigma] = dataset3Params(X, y, Xval, yval)
  ##EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
  ##where you select the optimal (C, sigma) learning parameters to use for SVM
  ##with RBF kernel
  ##   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
  ##   sigma. You should complete this function to return the optimal C and 
  ##   sigma based on a cross-validation set.
  ##

  t = [10.^[-2:1], 3*10.^[-2:1]];

  n = length (t);

  best = zeros (n);

  for i=1:n
    for j=1:n
      C = t(i);
      sigma = t(j);
      model = svmTrain (X, y, C, 
                        @(x1, x2) gaussianKernel (x1, x2, sigma),
                        1e-3, 20);
      pred = svmPredict (model, Xval);
      best(i,j) = mean (yval != pred);
    endfor
  endfor

  [~, Idx] = min (best(:));
  [i, j] = ind2sub ([n,n], idx);
  C = t(i);
  sigma = t(j);

endfunction
