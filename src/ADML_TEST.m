% clear context
clc
clear all

%parameter given in the article
C=1;
Gamma=[]; %linear kernel
% Gamma = 0.5; %uncomment for RBF kernel
nu=[];%0.1;
grid_size=10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ATTACK PARAMETERS
%step size given in the article
step = 0.15;
%step = 0.01;
%number of attack points and coordinate + label
num_points = 1;
yc = -1; % attacked class lable

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% DATASETS
%
%TR and TS samples/class
n=25; m=500;

%Gaussian DATA
mu=1.5; sigma = 0.6; num_feat=2;

%training data set
[x y] = load_gaussian_data(mu,sigma,n,num_feat);

% x_tst=x; % set the test data the same as the training data
% y_tst=y;  %
[x_tst y_tst] = load_gaussian_data(mu,sigma,m,num_feat);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% in case of one class SVM
if(~isempty(nu))
    C=nu; %we set this for convenience in passing the parameter to our functions
    x = x(y==+1,:);
    y = y(y==+1);
end
%trains SVM and evaluates its performance
[model alpha b SV_idx] = trainSVM(x,y,C,Gamma);
[~, score]=classifySVM(x_tst,y_tst,model);
[yclass score_update]=classifySVM(x,y,model);
err(1) = sum(yclass~=y)/size(x,1);

% margin_SV_idx = find(alpha > 1E-16 & alpha < C-1E-16);
% q=size(margin_SV_idx)
% Alfa=alpha(margin_SV_idx);

W_Coef = [];
[coefrow coefcol] = size(model.sv_coef);
for uu=1:coefrow
    if abs(model.sv_coef(uu)) ~= 0
        W_Coef = [W_Coef; model.sv_coef(uu)];
    end
end
T=model.SVs;
T=full(T);
SVs=T;
[c1, Px, PSVs] = intersect(x,SVs,'rows');
T=T([PSVs],:);
W_Coef = W_Coef([PSVs],:);
PL=y(Px);
ind=find(PL==yc);
T=T(ind,:);
PL=PL(ind);
W_Coef=W_Coef(ind);
[SVrow SVcol]=size(T);
TT=[];
%  for s=1:SVrow
s=1
init_points=T(s,:)
initial_alpha = W_Coef(s,:)
yc
[c1, Px, PSVs] = intersect(x,init_points,'rows');
x=x([1:end, Px],:);
y=y([1:end, Px],:);
%     y=[y; yc];
W_C_x_a = model.sv_coef'*full(model.SVs);
% attack point's position
% initial_alpha=Alfa(s); %
[xc_last err xc alpha_c xi_obj] = attackSVM(num_points, step, yc, x, y, x_tst, y_tst, C, Gamma,init_points,initial_alpha,W_C_x_a);
%  TT=[TT; xc_last]
%  end



