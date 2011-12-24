function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
  ##LEARNINGCURVE Generates the train and cross validation set errors needed 
  ##to plot a learning curve
  ##   [error_train, error_val] = ...
  ##       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
  ##       cross validation set errors for a learning curve. In particular, 
  ##       it returns two vectors of the same length - error_train and 
  ##       error_val. Then, error_train(i) contains the training error for
  ##       i examples (and similarly for error_val(i)).
  ##
  ##   In this function, you will compute the train and test errors for
  ##   dataset sizes from 1 up to m. In practice, when working with larger
  ##   datasets, you might want to do this in larger intervals.
  ##

  ## Number of training examples
  m = rows (X);

  ## Initialise outputs
  error_train = zeros(m, 1);
  error_val   = zeros(m, 1);

  ## It is not worth getting rid of this loop because the complexity is
  ## inside trainLinearReg which in turn is inside fmincg. While you
  ## *could* do some matrix gymnastics to perform a single minimisation
  ## in a much higher dimensional space instead of m minimimsations, the
  ## effort is unlikely to produce faster code.
  for i = 1:m
    Xtrain = X(1:i, :);
    ytrain = y(1:i);
    theta = trainLinearReg (Xtrain, ytrain, lambda);

    error_train(i) = sumsq (Xtrain*theta - ytrain)/(2*i);
    error_val(i) = sumsq (Xval*theta - yval)/(2*length (yval));
  endfor

endfunction
