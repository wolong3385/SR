%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function simulating an attack starting at a given points for SVMs 
% trained with a linear kernel.
% The attack is described at Algorithm 1 in the article Poisoning Attack
% Against SVMs.
% First a SVM is trained on the training data set plus the attack point
% Then the attack point is shifted in the feature space and its influence
% computed.
%
% Input parameters :
%   step : scalar representing the step size of the attack (gradient
%          method)
%   xc : row vector representing the initial position of the attack point
%   yc : integer which is the label associated to the attack point
%   x_tr : matrice of the training data set. Rows are observations and each
%          column is a feature defining data.
%   y_tr : column vector representing labels associated to training data
%   x_vd : matrice of the evaluation data set. Rows are observations and each
%          column is a feature defining data.
%   y_vd : column vector representing labels associated to evaluation data
%   C : scalar representing the cost incurred by the classifier when it
%       makes errors
%   initial_alpha : Value for statring attack point, This value is used for calculating the Gradient for the starting attack point
%   W_C_x_a : Initial weight regarding the attack point
% Output parameters :
%   xc : the last position computed by the attack
%   err : the number of errors performed by the classifier
%   xc_seq : matrice representing values taken by the attack point over the
%            attack
%   alphac_seq : column vector representing values that \alpha_c has
%                taken over the attack
%   xi_obj : column vector containing the value of the hinge loss function
%            at each iteration of the attack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xc err xc_seq alphac_seq xi_obj]= attack_linear_SVM(step, xc, yc, x_tr, y_tr, x_vd, y_vd, C,initial_alpha,W_C_x_a)
tt=0.5; % used for enlarge the w in order to shrink the margin
one_class = (numel(unique(y_tr)) == 1);
x_tr_ori = x_tr;
y_tr_ori = y_tr;
% train a SVM on the new training set and retrieve Support Vector indices
[model alpha] = trainSVM(x_tr,y_tr,C);
margin_SV_idx = find(alpha > 1E-26 & alpha < C-1E-26);
if(one_class)
    margin_SV_idx=find(alpha > 1E-26 & alpha < 1/size(x_tr,1) -1E-26);
end
    
% evaluation the performance of the trained SVM
[yclass score]=classifySVM(x_tr,y_tr,model);
err(1) = sum(yclass~=y_tr)/size(x_tr,1);
disp(['Validation error (%): ' num2str(100*err(1))]);

% compute the influence of the attack point on the errors
g = y_tr.*score -1;
if(one_class==1)
    g=g+1;
end
xi_obj(1) = sum(max(0,-g));

%update output variables
alphac_seq(1) = alpha(end);
xc_seq(1,:)=xc;

%kernel matrix for TR and VD
Q=lin(x_tr,x_tr).*(y_tr*y_tr');
Q_tst = lin(x_vd,x_tr).*(y_vd*y_tr');   
                              
councter_500=0;
alpha=initial_alpha  %%   added by kevin
G_w_old=[];
xi_obj_update_T = [];
   G_w=tt*W_C_x_a*initial_alpha*yc
for i=1:10
    councter_500=councter_500+1
    %update last row and last column
    Qic = lin(x_tr,xc).*(y_tr*yc);
    Q(:,end)=Qic';
    Q(end,:)=Qic;

    %compute attack direction and update the attack point's position
    dxc= attackSVMshift(x_tr,y_tr,x_tr(end,:),yc,margin_SV_idx,g,Q,Q_tst,x_vd,y_vd,alpha(end),G_w) 
    xc = xc + step*dxc;
    x_tr(end,:) = xc; 
    
%     if (i>2 & G_w~=G_w_old(i-1)) % terminates conditions added by kevin
%     else
%         xc=xc;
%         break
%     end 
%     xc_seq(i+1,:)=xc;
    
    % train a SVM on the updated training set and retrieve Support Vector indices
    [model alpha] = trainSVM(x_tr,y_tr,C);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

[yclass_update score_update]=classifySVM(x_tr_ori,y_tr_ori,model);
gg = y_tr_ori.*score_update -1;
if(one_class==1)
    gg=gg+1;
end
xi_obj_update = sum(max(0,-gg));
xi_obj_update_T = [xi_obj_update_T sum(max(0,-gg))];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    margin_SV_idx = find(alpha > 1E-6 & alpha < C-1E-6);
    if(one_class)
        margin_SV_idx=find(alpha > 1E-6 & alpha < 1/size(x_tr,1) -1E-6);
    end
    
    % evaluation the performance of the trained SVM
    [yclass score]=classifySVM(x_tr,y_tr,model);
    err(i+1) = sum(yclass~=y_tr)/size(x_tr,1);

    % compute the influence of the attack point on the errors
    g = y_tr.*score -1;
    if(one_class==1)
        g=g+1;
    end
    
    %update output variables
    xi_obj(i+1) = sum(max(0,-g));
    
    alphac_seq(i+1) = alpha(end);
    
   disp(['Validation error (%): ' num2str(100*err(i+1))]);
    % after 50 iterations, if the number of errors performed is less than a
    % threshold (10^-6), the attack is stopped
%     if(i>50 && err(i+1)-err(i-50) <= 1E-6)
%          return;
%     end

% the following is added by Kevin
     % $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ calculate gradient for Margin
    % G_alpha=alpha(margin_SV_idx)
    % G_y=y_tr(margin_SV_idx);
    % G=0.15*(G_alpha.*G_y);


% %     [c1, Px, PSVs] = intersect(full(model.SVs),xc,'rows'); % check if the attack point is SV or not
% %     if isempty(Px)
% %         w_x = 0;
% %     else
% %     w_x = model.sv_coef(Px);
% %     end
% %    

Coe = [];
[coerow coecol] = size(model.sv_coef);
for u=1:coerow
    if abs(model.sv_coef(u)) ~= 1
        Coe = [Coe; model.sv_coef(u)];
    end
end

    w_x = Coe(end);
    if 1-yc*(xc*w_x - model.rho)>= 0
        penal=1;
    else
        penal=0;
    end
alpha(end)
    G_w=tt*w_x*alpha(end)*yc*penal;
    G_w_old=[G_w_old G_w]
    % above is added by Kevin   $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%%%%%%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
[yclass score]=classifySVM(x_vd,y_vd,model);
    errt(i+1) = sum(yclass~=y_vd)/size(x_vd,1);
    disp(['Validation errort (%): ' num2str(100*errt(i+1))]);
%%%%%%%%%%%%%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
     
%     if (i>50 & xi_obj_update_T(i) == xi_obj_update_T(i-1)) % terminates conditions added by kevin
% %     else
% %         xc=xc;
%          break
%     else
%         return
%      end
    
end



