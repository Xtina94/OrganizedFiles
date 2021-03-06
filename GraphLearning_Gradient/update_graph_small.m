function [learned_Laplacian, learned_W] = update_graph_small(x, alpha, beta, maxEpoch, param,original_Laplacian, learned_W) 
%graph updating step by gradient descent

eye_N = eye(param.N_small);
for epoch = 1:maxEpoch
     %compute signal estimation
    [learned_dictionary, param] = construct_dict_small(param);
    estimated_y = learned_dictionary*x; 
    error(epoch) = sum(sum(abs(estimated_y-param.y_small))) + beta*sum(sum(abs(learned_W)));
    %computing the gradient
    K = max(param.K);
    der_all_new = zeros(param.N_small, param.N_small);
    learned_D = diag(sum(learned_W,2));
    learned_D_powers{1} = learned_D^(-0.5);
    learned_D_powers{2} = learned_D^(-1);
    for s=1:param.S
        for k=0:K
            C=zeros(param.N_small,param.N_small);
            B=zeros(param.N_small,param.N_small);
            for r=0:k-1 
                A = learned_D_powers{1}*param.Laplacian_powers_small{k-r}*x((s-1)*param.N_small+1:s*param.N_small,:)*(estimated_y - param.y_small)'*param.Laplacian_powers_small{r+1} * learned_D_powers{1};
                B=B+learned_D_powers{1}*learned_W*A*learned_D_powers{1};
                C=C-2*A';
                B=B+A*learned_W*learned_D_powers{2};
            end
            B = ones(size(B)) * (B .* eye_N);
            C = param.alpha{s}(k+1)*(C+B);
            der_all_new = der_all_new + C;
        end            
    end
    %adding the sparsity term gradient
    der_all_new = der_all_new +  beta*sign(learned_W); 
    %making derivative symmetric and removing the diag (that we don't want to change)
    der_sym = (der_all_new + der_all_new')/2 - diag(diag(der_all_new)); 
    
    %gradient descent, adjusting the weights with each step
    alpha = alpha * (0.1^(1/maxEpoch));
    %beta = beta * (10^(1/maxEpoch));
    learned_W = learned_W - alpha * der_sym;
    
    %producing a valid weight matrix
    learned_W(learned_W<0)=0;
    for i = 1:param.N
        if isempty(find(param.sources == i))
            for j = 1:param.N
                if isempty(find(param.sources == j))
                    learned_W(i,j) = 0;
                end
            end
        end
    end
    
    % combinatorial Laplacian
    learned_L = diag(sum(learned_W,2)) - learned_W;
    % normalized Laplacian
    param.Laplacian_small = (diag(sum(learned_W,2)))^(-1/2)*learned_L*(diag(sum(learned_W,2)))^(-1/2);
    % producing a valid Laplacian
    for i = 1:param.N
        if isempty(find(param.sources == i))
            param.Laplacian_small(:,i) = 0;
        end
    end
    
end
learned_Laplacian = param.Laplacian_small;
end

