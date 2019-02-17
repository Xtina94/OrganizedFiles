clear all
close all

%% Adding the paths
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\Optimizers'); %Folder conatining the yalmip tools
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\Comparison_datasets\'); %Folder containing the copmarison datasets
addpath('C:\Users\Cristina\Documents\GitHub\OrganizedFiles\DataSets\'); %Folder containing the training and verification dataset

%% Loaging the required dataset
flag = 2;
switch flag
    case 1
        load ComparisonDorina.mat
        load DataSetDorina.mat
    case 2
        load ComparisonHeat30.mat
        load DataSetHeat30.mat
        param.alpha = comp_alpha;
    case 3
        load ComparisonUber.mat
        load DataSetUber.mat
    case 4
        load ComparisonDoubleHeat.mat
        load DataSetDoubleHeat.mat
    case 5
        load ComparisonDorinaLF.mat
        load DataSetDorinaLF.mat
end

%% Set the parameters

switch flag
    case 1 %Dorina
        param.S = 4;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        param.N = 100; % number of nodes in the graph
        ds = 'Dataset used: Synthetic data from Dorina';
        ds_name = 'Dorina';
        param.percentage = 15;
        param.thresh = param.percentage+60;
    case 2 %1 LF heat kernel
        param.S = 1;
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 30;
        ds = 'Dataset used: Synthetic data from Dorina - 1 single kernel';
        ds_name = 'Heat';
        param.percentage = 8;
        param.thresh = param.percentage + 18;
    case 3 %Uber
        param.S = 2;
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 29; % number of nodes in the graph
        ds = 'Dataset used: data from Uber';
        ds_name = 'Uber';
        param.percentage = 8;
        param.thresh = param.percentage+6;
    case 4 %Cristina
        param.S = 2;  % number of subdictionaries 
        param.epsilon = 0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 15;
        param.N = 30; % number of nodes in the graph
        ds = 'Dataset used: data from double heat kernel';
        ds_name = 'DoubleHeat'; 
        param.percentage = 8;
        param.thresh = param.percentage+6;   
        temp = comp_alpha;
        comp_alpha = zeros((degree+1)*param.S,1);
        for i = 1:2
            comp_alpha((degree+1)*(i-1)+1:(degree+1)*i) = temp(:,i);
        end
    case 5
        param.S = 1;
        param.epsilon = 0.2; % we assume that epsilon_1 = epsilon_2 = epsilon
        degree = 20;
        param.N = 30;
        ds = 'Dataset used: Synthetic data from Dorina - 1 single kernel';
        ds_name = 'DorinaLF';
        param.percentage = 15;
        param.thresh = param.percentage + 13;
end

param.M = 5; % Number of known sources we impose
param.J = param.N * param.S; % total number of atoms 
param.K = degree*ones(1,param.S);
param.T0 = 4; % sparsity level in the training phase
param.c = 1; % spectral control parameters
param.mu = 1e-2; % polynomial regularizer paremeter
path = ['C:\Users\Cristina\Documents\GitHub\OrganizedFiles\GraphLearning\Results\13.07.18',num2str(ds_name),'\']; %Folder containing the results to save
    
%% Graph learning algorithm

param.displayProgress = 1;
param.numIteration = 50;
param.plot_kernels = 1; % plot thelearned polynomial kernels after each iteration
param.quadratic = 0; % solve the quadratic program using interior point methods

disp('Starting to train the graph structure');

[learned_dictionary,output]  = Graph_learning(TrainSignal,param);

CoefMatrix_Pol = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%% Compute the l-2 norms

lambda_norm = 'is 0 since here we are learning only the kernels'; %norm(comp_eigenVal - eigenVal);
% % % alpha_norm = norm(comp_alpha - output_Pol.alpha(1:(param.S - 1)*(degree + 1)));
% % % X_norm = norm(comp_X - CoefMatrix_Pol(1:(param.S - 1)*param.N,:));
% % % D_norm = norm(comp_D - Dictionary_Pol(:,1:(param.S - 1)*param.N));
alpha_norm = norm(comp_alpha - output_Pol.alpha);
X_norm = norm(comp_X - CoefMatrix_Pol);
tot_norm_X = norm([(comp_train_X - output_Pol.CoefMatrix) (comp_X - CoefMatrix_Pol)]);
D_norm = norm(comp_D - learned_dictionary);
W_norm = 'is 0 since here we are learning only the kernels';

%% Compare the learned coefficients
temp = output_Pol.alpha;
final_alpha = cell(param.S,1);
for i = 1:param.S
    final_alpha{i} = temp((k+1)*(i-1) + 1:(k+1)*i); 
end

figure('Name','The different behavior of the alpha coefficients')
for j = 1:param.S
    subplot(1,param.S,j)
    title(['Comp ker ',num2str(j),'  VS Learned ker ',num2str(j)]);
    hold on
    stem(comp_alpha(:,j))
    stem(final_alpha{j})
    hold off
    legend('comparison kernel','learned kernel');
end
filename = [path,'\AlphaCoeff_comparison','.png'];
saveas(gcf,filename);

%% Compute the average CPU_time

avgCPU = mean(output_Pol.cpuTime);

%% Save the results to file

% The norms
filename = [path,'Norms.mat'];
save(filename,'lambda_norm','alpha_norm','X_norm','D_norm','W_norm');

% The Output data
filename = [path,'Output.mat'];
learned_alpha = output_Pol.alpha;
% save(filename,'ds','Dictionary_Pol','learned_alpha','CoefMatrix_Pol','errorTesting_Pol','avgCPU','tot_norm_X');
save(filename,'errorTesting_Pol','avgCPU','tot_norm_X');

% The kernels plot

figure('Name','Comparison between the Kernels')
subplot(2,1,1)
title('Original kernels');
hold on
for s = 1 : param.S
    plot(comp_lambdaSym,comp_ker(:,s));
end
hold off
subplot(2,1,2)
title('learned kernels');
hold on
for s = 1 : param.S
%     plot(param.lambda_sym(4:length(param.lambda_sym)),output_Pol.kernel(4:length(output_Pol.kernel),s));
plot(param.lambda_sym,output_Pol.kernel(:,s));
end
hold off

filename = [path,'FinalKernels_plot.png'];
saveas(gcf,filename);

% The CPU time plot
xq = 0:0.2:param.numIteration;
figure('Name','CPU time per iteration')
vq2 = interp1(1:param.numIteration,output_Pol.cpuTime,xq,'spline');
plot(1:param.numIteration,output_Pol.cpuTime,'o',xq,vq2,':.');
xlim([0 param.numIteration]);

filename = [path,'AvgCPUtime_plot.png'];
saveas(gcf,filename);

