function p = predict(Theta1, Theta2, X)
  ##PREDICT Predict the label of an input given a trained neural network
  ##   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
  ##   trained weights of a neural network (Theta1, Theta2)

  m = rows (X);
  X = [ones(m, 1), X];

  ## See predictOneVsAll.m for an explanation of this syntax if it's new
  ## to you.
  [~, p] = max ( [ones(m, 1), sigmoid(X*Theta1')]*Theta2', [], 2);

endfunction
