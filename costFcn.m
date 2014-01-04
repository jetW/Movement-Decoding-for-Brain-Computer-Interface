function [F, G, H] = costFcn(Z,X,Y,Lambda,t)
% Compute the cost function F(Z)
%
% INPUTS: 
%   Z: Parameter values
% OUTPUTS
%   F: Function value
%   G: Gradient value
%   H: Hessian value
%
% @ 2011 Kiho Kwak -- kkwak@andrew.cmu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To improve the excution speed, please program your code with matrix
% format. It is 30 times faster than the code using the for-loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_data = size(X,2);
num_features = size(X,1);

%{
sym_kesi = sym('kesi',[1,num_data]);
sym_Lambda = sym('Lambda');
sym_W = sym('W',[num_features,1]);
sym_X = sym('X',[num_features,num_data]);
sym_Y = sym('Y',[1,num_data]);
sym_C = sym('C');
%}

%{
temp = sym_W' * sym_X;
temp = temp.*sym_Y;
temp = temp + sym_C*sym_Y + sym_kesi -1;
temp = -(1/t) * sum(log(temp));

temp1 = -(1/t) * sum(log(sym_kesi));

temp2 = sum(sym_kesi) + sym_Lambda * (sym_W' * sym_W);

F(sym_kesi,sym_Lambda,sym_W,sym_X,sym_Y,sym_C)= temp + temp1 + temp2;
%}

     



W = Z(1:num_features);
C = Z(num_features+1);
kesi = Z(num_features+2:num_features+num_data+1);


W = W';
kesi = kesi';
%calculate W' * X 

temp = Y.*(W'*X) + C*Y + kesi' - 1;
temp1 = -(1/t) * sum(log10(temp));

temp2 = (1/t) * sum(log10(kesi));

temp3 = sum(kesi) + Lambda * (W' * W);

F= temp1 - temp2 + temp3;

%calculate gradient of W
temp5 = Y./temp;
gradient_W = zeros(1,num_features);
for i = 1:num_features,
    gradient_W(i) = 2 * Lambda * W(i) - (1/t) * sum(X(i,:).*temp5);
end

%calculate gradient of C
gradient_C = -(1/t) * sum(Y./temp);

%calculate gradient of kesi
gradient_kesi = zeros(1,num_data);
for i = 1:num_data,
    gradient_kesi(i) = 1 - (1/t) * 1/((W'*X(:,i))*Y(i)+C*Y(i)+kesi(i)-1)-(1/t)*(1/kesi(i));
end

gradient = [gradient_W,gradient_C,gradient_kesi];
G = gradient;
%calculate hessian for W
temp6 = temp5.*Y;
temp7 = temp6./temp;
temp8 = temp5.^2;
Hessian = zeros(num_features+1+num_data,num_features+1+num_data);
for i = 1:num_features,
    for j = 1:size(Hessian,2),
        if(j==i),
            Hessian(i,j) = 2*Lambda + (1/t) * sum(temp8.*(X(i,:).^2));       
        elseif(j>i && j<=num_features),
            Hessian(i,j) = (1/t) * sum(temp8.*(X(i,:).*X(j,:)));
        elseif(j == num_features + 1),           
            Hessian(i,j) = (1/t) * sum((X(i,:).*temp8));      
        elseif(j>num_features+1),
            index = j - (num_features+1);
            Hessian(i,j) = (1/t) * X(i,index)*Y(index)/((temp(index))^2);
        else
            Hessian(i,j) = Hessian(j,i);
        end                
    end
end

%calculate hessian for C

for i = 1:num_features,
    Hessian(1+num_features,i) = Hessian(i,1+num_features);
end
Hessian(num_features+1,num_features+1) = (1/t) * sum(temp8);
for i = num_features+2:size(Hessian,2),
    index = i - (num_features+1);
    Hessian(num_features+1,i) = (1/t) * Y(index)/ (temp(index)^2);
end

for i = num_features+2:size(Hessian,1),
    for k = 1:size(Hessian,2),
        if(k<=num_features+1),
            Hessian(i,k) = Hessian(k,i);
        end
    end
end

%calculate hessian for kesi
Hessian_kesi = zeros(num_data,num_data);
for i = 1:num_data,
    for j = 1:num_data,
        if(j<i),
            Hessian_kesi(i,j) = Hessian_kesi(j,i);
        elseif(j==i),
            Hessian_kesi(i,i) = (1/t)*(1/(temp(j)^2)) + (1/t) * 1/(kesi(i)^2);
        else
            Hessian_kesi(i,j) = 0;
        end
    end
end

Hessian(num_features+2:end,num_features+2:end) = Hessian_kesi;
H = Hessian;



