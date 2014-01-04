function [optSolution, err] = solveOptProb_NM(costFcn,init_Z,Lambda,t,X,Y,tol)
% Compute the optimal solution using Newton method
%
% INPUTS:
%   costFcn: Function handle of F(Z)
%   init_Z: Initial value of Z
%   tol: Tolerance
%
% OUTPUTS:
%   optSolution: Optimal soultion
%   err: Errorr
%
num_feature = size(X,1);
num_data = size(X,2);

Z = init_Z;
err = 1;   

% Set the error 2*tol to make sure the loop runs at least once
flag = 1;
while (err/2) > tol
    if (flag == 1)
        flag = 0;
    else
        %backtracking line search to obtain s
        s = 1;
        while(true),
            temp_haha = Z + s * deltaZ';
            temp_W = temp_haha(1:num_feature);
            temp_C = temp_haha(num_feature+1);
            temp_kesi = temp_haha(num_feature+2:num_feature+1+num_data);
            temp = temp_W * X;
            temp1 = temp.* Y + temp_C * Y + temp_kesi - 1;
            if(isempty((find(temp1<=0,1))) && isempty((find(temp_kesi <=0,1)))),
                break;
            else
                s = 0.5 * s;
            end           
        end
        Z = Z + s*deltaZ'; 
    end
    [F, G, H] = feval(costFcn,Z,X,Y,Lambda,t);
    %compute the Newton step and decrement
    deltaZ = -inv(H) * G';
    delta = -(G * deltaZ);
    err= delta;
 end
 optSolution = Z;



