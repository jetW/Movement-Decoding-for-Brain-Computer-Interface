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
optLambda = 0;
minError = Inf;
optional_Lambda = [0.01,1,100,10000]; 

c1 = X(:,1:100);
c2 = X(:,101:200);
Label_1 = Y(1:100);
Label_2 = Y(101:200);
for j = 1:4 
    Lambda = optional_Lambda(j);
    error = 0.0;
    %five fold CV for a certain Lambda
    for i = 1:5,
     
        test_Index = (i-1)*20+1:(i-1)*20+20;
        train_Index = setdiff(1:100,test_Index);
        train_Data = [c1(:,train_Index),c2(:,train_Index)];
        test_Data = [c1(:,test_Index),c2(:,test_Index)];
        train_Label = [Label_1(train_Index),Label_2(train_Index)];
        test_Label = [Label_1(test_Index),Label_2(test_Index)];
        for k = 1:size(train_Data,2),
             zeta(k) = max(1-train_Label(k)*(setPara.W'*train_Data(:,k)+setPara.C),0) + 0.001;
        end
         init_Z = [W',C,zeta];
         test_value = train_Label.*(W'*train_Data) + C + zeta;
        t = setPara.t;
        while (t <= Tmax)    
            [optSol, err] = solveOptProb_NM(@costFcn, init_Z,Lambda,t,train_Data,train_Label,tol);
            init_Z = optSol;
            t = t * 15;
        end
        temp_W = optSol(1:num_Feature);
        temp_C = optSol(num_Feature+1);
        predict =  (temp_W * test_Data + temp_C);
        decision = predict.* test_Label;
        error = error + sum((decision<0));
    end
    if(error<minError),
        minError = error;
        optLamda = Lambda;
        optSolution = optSol;
    end
end
