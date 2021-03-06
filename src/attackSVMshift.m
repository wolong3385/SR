% This code is for non-commercial use.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the direction of the attack to make the classifier performs as
% much errors as possible. This function only applies in the case of the
% use of a linear kernel.
%
% Input parameters :
%   x : a matrix containing the training support vectors. A row represents
%   an observation and each column would be a feature of the data.
%   y : column vector representing labels associated to training data
%   xc : row vector representing the initial position of the attack point
%   yc : integer which is the label associated to the attack point
%   margin_SV_idx : a column vector containing the indices of points which
%   are on the margins of the solution given by the computed SVM. (Support
%   vectors are points for which their alpha value is > 0 and < C
%   g : a column vector representing the score of the class assigned to an
%       example
%   Q : a matrice representing the gradient of the linear kernel. This matrix
%       has a size of size(x)*size(x) 
%   Q_tst :a matrice representing the gradient of the linear kernel. This
%          matrix has a size of size(x_tst)*size(x)
%   x_tst : matrice of the evaluation data set. Rows are observations and each
%           column is a feature defining data.
%   y_tst : column vector representing labels associated to evaluation data
%   alphac : a column vector containing the alpha value of the attack point
% Output parameters :
%   dxc : a vector representing the direction of the attack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dxc  = attackSVMshift(x,y,xc,yc,margin_SV_idx,g,Q,Q_tst,x_tst,y_tst,alphac,G_w)

% if there is no support vectors on the margins of the previous solution of
% SVM, do not move the attack point
if(isempty(margin_SV_idx))
    dxc=zeros(1,size(xc,2));
    return;
end
Q = Q+1E-6*eye(size(Q,1));
R = [0 y(margin_SV_idx)'; y(margin_SV_idx) Q(margin_SV_idx,margin_SV_idx)]^-1;
% Xi = repmat(y_tst,1,size(x,2)).*x_tst; %this is to have y_i * x_i  % commented by kevin
Xi = repmat(y,1,size(x,2)).*x; %this is to have y_i * x_i   % added by kevin for Qkc
Xs = repmat(y(margin_SV_idx),1,size(x,2)).*x(margin_SV_idx,:); %y_s * x_s
delta = -R*[zeros(1,size(Xs,2)); Xs];
disp(delta);
%db and da need to be multiply by alpha_c to match their definition in the
%article
db = delta(1,:);
da = delta(2:end,:);
Qis = Q_tst(:,margin_SV_idx);
% delta_gi = (Qis*da + y_tst*db + Xi)*yc*alphac;  %commented by kevin
delta_gi = (Qis*da + y*db + Xi)*yc*alphac; % added by kevin
delta_gi(g>=0,:)=0; %slack variables...
gradient_W = -G_w; % by Kevin
gradient = -sum(delta_gi);
gradient=gradient+gradient_W;  %kevin
dxc =  gradient / norm(gradient);
disp(dxc);
%if normalization did not go well, do not move the attack point
if(isnan(dxc))
    dxc=zeros(1,size(xc,2));
    return;
end

