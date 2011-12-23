% Plot Data
plotData(X(:,2:3), y);

hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
  
test_lamda = [0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100, 300];
   
for iter=1:length(test_lamda)

    % Initialize fitting parameters
    initial_theta = zeros(size(X, 2), 1);

    % Set regularization parameter lambda to 1 (you should vary this)
    

    
    lambda = test_lamda(iter);

    % Set Options
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Optimize
    [theta, J, exit_flag] = ...
        fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

    % Plot Boundary
%     plotDecisionBoundary(theta, X, y);


    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
    

%     hold off;

    % Compute accuracy on our training set
    p = predict(theta, X);

    fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
    
end

hold off