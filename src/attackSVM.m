%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function runs the gradient-based attack for num_points different points and
%returns the *best* run only.
% Input parameters :
%   num_points : integer representing the number of points of attack
%   step : scalar defining the step size (gradient method)
%   yc : integer representing the class of the attack point

%   x : matrice of the training data set. Rows are observations and each
%       column is a feature defining data.
%   y : column vector representing labels associated to training data
%   x_tst : matrice of the evaluation data set. Rows are observations and each
%           column is a feature defining data. In this study, x_tst is the
%           same as x
%   y_tst : column vector representing labels associated to evaluation
%   data, In this study, y_tst is the
%           same as y
%   C : scalar representing the cost incurred by the classifier when it
%       makes errors
%   Gamma : scalar representing the gamma value used in RBF kernels
%   init_points : matrice containing num_points on which the attack starts.
%                 Rows are observations and each column is a feature defining data.
%   initial_alpha : Value for statring attack point, This value is used for calculating the Gradient for the starting attack point
%   W_C_x_a : Initial weight regarding the attack point
% Output parameters :
%   xc_last : a row vector containing the last position of the best attack point
%   err : a column vector containing the number of errors performed by
%         the classifier at each iteration of the simulated attack for the best
%         attack point
%   xc : a matrice containing the position of the attack point 
%        at each iteration of the simulated attack.
%        Each row represents an iteration and each column a feature.
%   alpha_c : a column vector containing the alpha value of the best attack
%             point for each iteration of the simulated attack
%   xi_obj : column vector containing the value of the hinge loss function
%            at each iteration of the simulated attack for the best attack
%            point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xc_last err xc alpha_c xi_obj] = attackSVM(num_points, step, yc, x, y, x_tst, y_tst, C, Gamma,init_points,initial_alpha,W_C_x_a)



% if no starting points have been given
if (nargin < 12)
    %choose a random point from opposite class -yc to start with...
    selection_criteria = (y_tst==-yc);
    % choose the index randomly and uniformly in the selection_criteria
    % subset
    r = randsample(sum(selection_criteria),num_points);
    %build a subset of data
    init_points = x_tst(selection_criteria,:);
    %pick a starting point
    init_points = init_points(r,:);
end

err = -inf;

% for each attack point, compute the influence on the performance of the
% classifier and keep in memory the best point of attack
for i=1:num_points
        [xc_last_i err_i xc_i alpha_c_i xi_obj_i]= attack_linear_SVM(step,init_points(i,:), yc, x, y, x_tst, y_tst, C,initial_alpha,W_C_x_a);   
    if(max(err_i) > max(err))
        xc_last = xc_last_i;
        err = err_i;
        xc = xc_i;
        alpha_c = alpha_c_i;
        xi_obj = xi_obj_i;
    end
               
        
end