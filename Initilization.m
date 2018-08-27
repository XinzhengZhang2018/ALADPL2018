function [ DataMat, DictMat, P_Mat, DataInvMat, CoefMat ] = Initilization( Data , Label, DictSize, tau, lambda, gamma )
% In this intilization function, we do the following things:
% 1. Random initialization of dictioanry pair D and P for each class
% 2. Precompute the class-specific inverse matrix used in Eq. (10)
% 3. Compute matrix class-specific code matrix A by Eq. (8) 
%    with the random initilized D and P
% 
% The randn seeds are setted to make sure the results in our paper are
% reproduceable. The randn seed setting can be removed, our algorithm is 
% not sensitive to the initilization of D and P. In most cases, different 
% initilization will lead to the same recognition accuracy on a wide randge
% of testing databases.


ClassNum = max(Label);
Dim      = size(Data,1);%维数
I_Mat    = eye(Dim,Dim);%对角阵

for i=1:ClassNum
    TempData      = Data(:,Label==i);%i类测试样本
    DataMat{i}    = TempData;
    randn('seed',i); %相同的seed对应startpoint相同，产生相同的随机序列                       
    DictMat{i}    = normcol_equal(randn(Dim, DictSize));%初始D
    randn('seed',2*i);
    P_Mat{i}      = normcol_equal(randn(Dim, DictSize))';%初始P
 
    TempDataC     = Data(:,Label~=i);%除i类所有的样本
    DataInvMat{i} = inv(tau*TempData*TempData'+lambda*TempDataC*TempDataC'+gamma*I_Mat);
end

CoefMat = UpdateA(  DictMat, DataMat, P_Mat,  tau, DictSize  );

