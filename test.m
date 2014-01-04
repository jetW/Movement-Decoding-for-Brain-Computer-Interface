load('feaSubEImg.mat');
%load('feaSubEovert.mat');
X1 = [class{1},class{2}];
Label1 = [ones(1,120),-ones(1,120)]';
num_Data = size(Label1,1);
num_Feature = size(X1,1);
index = 1:num_Data;
Data = X1(:,index);
Label = Label1(index)';
num_Fold = 6;
c1 = X1(:,1:120);
c2 = X1(:,121:240);
Label_1 = Label(1:120);
Label_2 = Label(121:240);
for i = 1:6,
    setPara.W = ones(num_Feature,1);
    setPara.C = 0;
    setPara.t = 1;
    setPara.tol = 0.000001;
    setPara.Tmax = 1000000;
    %setPara.zeta = ones((num_Fold-1)*num_Data/num_Fold,1);
    test_Index = (i-1) * 20 +1 :(i-1)*20+20;    
    train_Index = setdiff(1:120,test_Index);
    train_data = [c1(:,train_Index),c2(:,train_Index)];
    train_label = [Label_1(train_Index),Label_2(train_Index)];
    test_data = [c1(:,test_Index),c2(:,test_Index)];
    test_label = [Label_1(test_Index),Label_2(test_Index)];
    %for j = 1:size(train_data,2),
     %   zeta(j) = max(1-train_label(j)*(setPara.W'*train_data(:,j)+setPara.C),0) + 0.001;
    %end
    %setPara.zeta = zeta;
    %value = train_label.*(setPara.W'*train_data) + setPara.C + setPara.zeta;
    [solution,optLamda] = getOptLamda(train_data,train_label,setPara);
    W = solution(1:204);
    C = solution(205);
    predict = W*test_data + C;
    decision = predict.* test_label;
    correct = sum(decision > 0);
    percentage = correct/40;
    disp(percentage);
    AC(i) = percentage;
end
