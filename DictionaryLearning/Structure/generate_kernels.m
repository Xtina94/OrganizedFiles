function output = generate_kernels(param)
% Generates the kernels' polynomial coefficients in a random way
% input:
%       degree = polynomial degree
%       n_kernels = number of different kernels
%       lambdas = vector of eigenvalues
%       n_roots = number of lambdas which are roots of the polynomial

S = param.S;
K = max(param.K);
% % % m = param.percentage;

coefficients = zeros(S,K+1);
lambda_powers = param.lambda_power_matrix;

for i = 1:S
    for j = 1:length(coefficients(i,:))
        coefficients(i,j) = ((1)^(randi(10,1))*rand(1));
    end
end
output.coefficients = coefficients';
output.kernels = zeros(param.N,param.S);
for s = 1:S
    for i = 1:param.N
        output.kernels(i,s) = (lambda_powers(i,:)*output.coefficients(:,s));
    end
end

%% Plotting the kernels
figure('Name', 'Original Generated Kernels')
for s = 1:S
    hold on;
    plot(lambda_powers(:,2),output.kernels(:,s));
end
hold off;
end