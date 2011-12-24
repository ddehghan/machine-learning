function [J grad] = nnCostFunction(nn_params, 
                                   input_layer_size,
                                   hidden_layer_size,
                                   num_labels,
                                   X, y, lambda)
  ##NNCOSTFUNCTION Implements the neural network cost function for a two layer
  ##neural network which performs classification
  ##   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  ##   X, y, lambda) computes the cost and gradient of the neural network. The
  ##   parameters for the neural network are "unrolled" into the vector
  ##   nn_params and need to be converted back into the weight matrices. 
  ## 
  ##   The returned parameter grad should be a "unrolled" vector of the
  ##   partial derivatives of the neural network.
  ##

  ## Reshape nn_params back into the parameters Theta1 and Theta2, the
  ## weight matrices for our 2 layer neural network
  Theta1 = reshape (nn_params(1:hidden_layer_size * (input_layer_size + 1)), 
                    hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape (nn_params((1 + (hidden_layer_size
                                    * (input_layer_size + 1))):end),
                    num_labels, (hidden_layer_size + 1));

  ## Setup some useful variables
  m = rows (X);
  one_vec = ones (m, 1);

  a1 = X;
  z2 = [one_vec, a1]*Theta1';
  a2 = sigmoid (z2);
  z3 = [one_vec, a2]*Theta2';
  a3 = sigmoid (z3);
  ht = a3;

  ## Logical matrix of zeros and ones representing the labels
  y_idx = bsxfun (@eq, 1:num_labels, y);

  ## Using long form of cost function that broke it up into cases.
  J = -(sum (log (ht(y_idx))) + sum (log (1 - ht(! y_idx))))/m     \
      
      ## The regularisation term has to exclude the first column of the Thetas,
      ## because we don't regularise the bias nodes.
      + lambda*(sumsq (Theta1(:, 2:end)(:))                         \
                + sumsq (Theta2(:, 2:end)(:)))/(2*m);

  ## Backprop
  delta3 = a3 - y_idx;
  delta2 = (delta3*Theta2)(:, 2:end) .* sigmoidGradient (z2);

  Theta2_grad = delta3' * [one_vec, a2] / m;
  Theta1_grad = delta2' * [one_vec, a1] / m;

  ## Add regularisation terms
  Theta2_grad(:, 2:end) += Theta2(:, 2:end)*lambda/m;
  Theta1_grad(:, 2:end) += Theta1(:, 2:end)*lambda/m;

  grad = [Theta1_grad(:) ; Theta2_grad(:)];

endfunction
