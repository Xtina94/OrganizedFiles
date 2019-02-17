function alpha_coefficients = polynomial_construct_high(param)

K = max(param.K);
m = param.percentage;
q = sum(param.K)+param.S;

alpha = sdpvar(K+1,1);

% % %     gamma = gammaCoeff;
% % %     gamma_2 = gammaCoeff;
% % %     gamma_3 = gammaCoeff;
% % %     gamma_4 = gammaCoeff;

beta = param.beta_coefficients;

%% First part of the vector

% For block A
for i = 1:K-m+1
    alpha(i,1) = gamma(1,1) *  beta(i,1);
    for j = 2:i
        alpha(i,1) = alpha(i,1) + gamma(j,1) * beta(i-j+1,1);
    end
end

% For block B
for i = K-m+2:m+1
    alpha(i,1) = gamma(1,1) * beta(i,1);
    for j = 2:K-m+1
        alpha(i,1) = alpha(i,1) + gamma(j,1) * beta(i-j+1,1);
    end
end

% For block C
index = 2;
for i = m+2:K+1
    alpha(i,1) = gamma(index,1) * beta(m+1,1);
    for j = index+1:K-m+1
        alpha(i,1) = alpha(i,1) + gamma(j,1) * beta(m+1-(j-index),1);
    end
    index = index + 1;
end

%% Second part of the vector

% For block A
for i = 1:K-m+1
    alpha_2(i,1) = gamma_2(1,1) *  beta(i,1);
    for j = 2:i
        alpha_2(i,1) = alpha_2(i,1) + gamma_2(j,1) * beta(i-j+1,1);
    end
end

% For block B
for i = K-m+2:m+1
    alpha_2(i,1) = gamma_2(1,1) * beta(i,1);
    for j = 2:K-m+1
        alpha_2(i,1) = alpha_2(i,1) + gamma_2(j,1) * beta(i-j+1,1);
    end
end

% For block C
index = 2;
for i = m+2:K+1
    alpha_2(i,1) = gamma_2(index,1) * beta(m+1,1);
    for j = index+1:K-m+1
        alpha_2(i,1) = alpha_2(i,1) + gamma_2(j,1) * beta(m+1-(j-index),1);
    end
    index = index + 1;
end

%% Third part of the vector
if param.S == 4
    alpha_3 = sdpvar(K+1,1);
    alpha_4 = sdpvar(K+1,1);
    gamma_3 = sdpvar(K-m+1,1);
    gamma_4 = sdpvar(K-m+1,1);
    
    % For block A
    for i = 1:K-m+1
        alpha_3(i,1) = gamma_3(1,1) *  beta(i,1);
        for j = 2:i
            alpha_3(i,1) = alpha_3(i,1) + gamma_3(j,1) * beta(i-j+1,1);
        end
    end
    
    % For block B
    for i = K-m+2:m+1
        alpha_3(i,1) = gamma_3(1,1) * beta(i,1);
        for j = 2:K-m+1
            alpha_3(i,1) = alpha_3(i,1) + gamma_3(j,1) * beta(i-j+1,1);
        end
    end
    
    % For block C
    index = 2;
    for i = m+2:K+1
        alpha_3(i,1) = gamma_3(index,1) * beta(m+1,1);
        for j = index+1:K-m+1
            alpha_3(i,1) = alpha_3(i,1) + gamma_3(j,1) * beta(m+1-(j-index),1);
        end
        index = index + 1;
    end
    
    %% Fourth part of the vector
    
    % For block A
    for i = 1:K-m+1
        alpha_4(i,1) = gamma_4(1,1) *  beta(i,1);
        for j = 2:i
            alpha_4(i,1) = alpha_4(i,1) + gamma_4(j,1) * beta(i-j+1,1);
        end
    end
    
    % For block B
    for i = K-m+2:m+1
        alpha_4(i,1) = gamma_4(1,1) * beta(i,1);
        for j = 2:K-m+1
            alpha_4(i,1) = alpha_4(i,1) + gamma_4(j,1) * beta(i-j+1,1);
        end
    end
    
    % For block C
    index = 2;
    for i = m+2:K+1
        alpha_4(i,1) = gamma_4(index,1) * beta(m+1,1);
        for j = index+1:K-m+1
            alpha_4(i,1) = alpha_4(i,1) + gamma_4(j,1) * beta(m+1-(j-index),1);
        end
        index = index + 1;
    end
    
    alpha_coefficients = [alpha' alpha_2' alpha_3' alpha_4']';
else
    alpha_coefficients = [alpha' alpha_2']';
end
end
