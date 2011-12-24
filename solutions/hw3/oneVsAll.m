function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  ##  [all_theta] = ONEVSALL(X, y, num_labels, lambda)
  ##
  ##   trains num_labels logisitc regression classifiers and returns
  ##   each of these classifiers in a matrix all_theta, where the i-th
  ##   row of all_theta corresponds to the classifier for label i

  m = rows (X);
  n = columns (X);

  all_theta = zeros(num_labels, n + 1);
  X = [ones(m, 1) X];

  for c = 1:num_labels

    ## The following uses @(x) ... notation for defining an anonymous
    ## function (also known as a closure or a lambda) to be fed to
    ## fmincg
    all_theta(c, :) = fmincg ( @(t) lrCostFunction (t, X, y == c, lambda),
                              zeros (n+1, 1));
  endfor

endfunction
