function [optSolution,optLamda] = getOptLamda(X, Y, setPara)
% Get the optimal lamda
%
% INPUTS:
%   X(MxN) : trData(i,j) is the i-th feature from the j-th trial
%   Y(Nx1): trData(j) is the label of the j-th trial (1 or -1)
%   setPara : Initialized parameters
%            setPara.t      
%            setPara.zeta   
%            setPara.Tmax   
%            setPara.tol    
%            setPara.W      
%            setPara.C      
%
% OUTPUTS:
%   optiLamda: Optimal lamda value 
%
num_Data = size(X,2);
num_Feature = size(X,1);
W = setPara.W;
C = setPara.C;
t = setPara.t;
tol = setPara.tol;
Tmax = setPara.Tmax;
zeta = setPara.zeta;

init_Z = [W, C, zeta];
optLambda = 0;
minError = Inf;
optional_Lambda = [0.01,1,100,10000]; 
for j = 1:4 
    Lambda = optional_Lambda(j);
    error = 0.0;
    for i = 1:5,
        test_Index = (i-1)*num_Data/4+1:(i-1)*num_Data/4+numData/4;
        train_Index = [1:num_Data] - test_Index;
        train_Data = X(:,train_Index);
        test_Data = X(:,test_Index);
        train_Label = Y(train_Index);
        test_Label = Y(test_Index);

        while (t <= Tmax)    
            [optSolution, err] = solveOptProb_NM(@costFcn, init_Z,Lambda,tol,t,train_Data,train_Label,tol);
            t = t * 15;
        end
        temp_W = optSolution(1:num_Feature);
        temp_C = optSolution(num_Feature+1);
        predict =  (temp_W' * test_Data + temp_C);
        decision = predict.* test_Label;
        error = error + sum((decision<1));
    end
    if(error<minError),
        minError = error;
        optLambda = Lambda;
    end
    disp(minError);
end
