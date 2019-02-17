function [recovered_Dictionary,output,g_ker] = Polynomial_Dictionary_Learning(Y, param)

% Set parameters

lambda_sym = param.lambda_sym;
lambda_powers = param.lambda_powers;
Laplacian_powers = param.Laplacian_powers;

if (~isfield(param,'displayProgress'))
    param.displayProgress = 0;
end

if (~isfield(param,'quadratic'))
    param.quadratic = 0;
end

if (~isfield(param,'plot_kernels'))
    param.plot_kernels = 0;
end

if (~isfield(param,'numIteration'))
    param.numIteration = 100;
end

if (~isfield(param,'InitializationMethod'))
    param.InitializationMethod = 'Random_kernels';
end

color_matrix = ['b', 'r', 'g', 'c', 'm', 'k', 'y'];
 
% Initializing the dictionary

if (strcmp(param.InitializationMethod,'Random_kernels')) 
    [Dictionary(:,1 : param.J)] = initialize_dictionary(param);
   
    
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
        Dictionary(:,1 : param.J) = param.initialDictionary(:,1 : param.J);  %initialize with a given initialization dictionary
else 
    display('Initialization method is not valid')
end


% Graph dictionary learning algorithm

cpuTime = zeros(1,param.numIteration);
for iterNum = 1 : param.numIteration

    % OMP step
      CoefMatrix = OMP_non_normalized_atoms(Dictionary,Y, param.T0);
    
    % Dicitonary update step
          
       if (param.quadratic == 0)
           if (iterNum == 1)
            disp('solving the quadratic problem with YALMIP...')
           end
            [alpha, diagnostics] = coefficient_update_interior_point(Y,CoefMatrix,param,'sdpt3');
            cpuTime(iterNum) = diagnostics.solveroutput.info.cputime;
       else
           if (iterNum == 1)
            disp('solving the quadratic problem with ADMM...')
           end
            [Q1,Q2, B, h] = compute_ADMM_entries(Y, param, Laplacian_powers, CoefMatrix);
             alpha = coefficient_upadate_ADMM(Q1, Q2, B, h);
       end
    
        
        if (param.plot_kernels == 1) 
            g_ker = zeros(param.N, param.S);
            r = 0;
            for i = 1 : param.S
                for n = 1 : param.N
                p = 0;
                for l = 0 : param.K(i)
                    p = p +  alpha(l + 1 + r)*lambda_powers{n}(l + 1);
                end
                g_ker(n,i) = p;
                end
                r = sum(param.K(1:i)) + i;
            end
        end
        r = 0;
        for j = 1 : param.S
            D = zeros(param.N);
            for ii = 0 : param.K(j)
                D = D +  alpha(ii + 1 + r) * Laplacian_powers{ii + 1};
            end
            r = sum(param.K(1:j)) + j;
            Dictionary(:,1 + (j - 1) * param.N : j * param.N) = D;
        end

    if (iterNum>1 && param.displayProgress)
             output.totalError(iterNum - 1) = sqrt(sum(sum((Y-Dictionary * CoefMatrix).^2))/numel(Y));
             disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalError(iterNum-1))]);
    end
    

end

output.cpuTime = cpuTime;
output.CoefMatrix = CoefMatrix;
output.alpha =  alpha;
recovered_Dictionary = Dictionary;

function Initial_Dictionary = initialize_dictionary(param)

%======================================================
   %%  Dictionary Initialization
%======================================================


%% Input:
%         param.N:        number of nodes of the graph
%         param.J:        number of atoms in the dictionary 
%         param.S:        number of subdictionaries
%         param.eigenMat: eigenvectors of the graph Laplacian
%         param.c:        upper-bound on the spectral representation of the kernels 
%           
%% Output: 
%         Initial_Dictionary: A matrix for initializing the dictionary
%======================================================


J = param.J;
N = param.N;
S = param.S;
c = param.c;
Initial_Dictionary = zeros(N,J);

for i = 1 : S
   
    tmpLambda = c * rand(param.N);
       
    if isempty(tmpLambda)
        disp('Initialization fails');
        exit;
    end
    
    tmpLambda = diag(tmpLambda(randperm(N)));
    Initial_Dictionary(:,1 + (i - 1) * N : i * N) = param.eigenMat * tmpLambda * param.eigenMat';
end



function [Q1,Q2, B, h] = compute_ADMM_entries(Y, param, Laplacian_powers, X)
%======================================================
   %% Compute the required entries for ADMM
%======================================================
% Description: Find Q1, Q2, B, h such that the quadratic program is
% expressed as: 
%       minimize     (1/2)*alpha'*Q1*alpha-Q2*alpha
%       subject to   B*alpha<=h
%======================================================

N = param.N;
c = param.c;
epsilon = param.epsilon;
mu = param.mu;
S = param.S;
K = max(param.K);
Lambda = param.lambda_power_matrix;



B1 = sparse(kron(eye(S),Lambda));
B2 = kron(ones(1,S),Lambda);

Phi = zeros(S*(K+1),1);
for i = 1 : N
         r = 0;
        for s = 1 : S
            for k = 0 : K
                Phi(k + 1 + r,(i - 1)*size(Y,2) +1 : i*size(Y,2)) = Laplacian_powers{k + 1}(i,:)*X((s-1)*N + 1 : s*N,1 : end);
            end
            r = sum(param.K(1 : s)) + s;
        end
end

YPhi = (Phi*vec(Y'))';
PhiPhiT = Phi*Phi';

Q2 = YPhi;
Q1 = PhiPhiT + mu*eye(size(PhiPhiT,2));


B = [B1; -B1; B2; -B2];
h = [c*ones(size(B1,1),1);zeros(size(B1,1),1);(c + epsilon)*ones(size(B2,1),1); -(c - epsilon)*ones(size(B2,1),1)];







