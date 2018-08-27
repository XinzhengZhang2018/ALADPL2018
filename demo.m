%% A demo for adaptive local aspect dictionary pair learning method 
close all;
clear all;
warning off
load('dataSet.mat');% testSet trainSet
load('TrLabel90.mat');%TrLabel90 train set label classified by classes90 aspect interval
trls=[ones(1,699) 2.*ones(1,699) 3.*ones(1,699)];%train set label classified by classes
ttls=[ones(1,588) 2.*ones(1,588) 3.*ones(1,588)];%test set label classified by classes
 

lamda = 0.1;
DictSize = 50; 
tau    = 0.01;
lambda = 0.001;
gamma  = 0.0001;

tic
%% Training dictionary
class=1
trainSetNew = zeros(size(trainSet)); 
trainSetNew(:,TrLabel90==class)=trainSet(:,TrLabel90==class);
[ DictMat1 , EncoderMat1 ] = TrainDPL(trainSetNew, trls, DictSize, tau, lambda, gamma);
class=2
trainSetNew = zeros(size(trainSet)); 
trainSetNew(:,TrLabel90==class)=trainSet(:,TrLabel90==class);
[ DictMat2 , EncoderMat2 ] = TrainDPL(trainSetNew, trls, DictSize, tau, lambda, gamma);
class=3
trainSetNew = zeros(size(trainSet)); 
trainSetNew(:,TrLabel90==class)=trainSet(:,TrLabel90==class);
[ DictMat3 , EncoderMat3 ] = TrainDPL(trainSetNew, trls, DictSize, tau, lambda, gamma);
class=4
trainSetNew = zeros(size(trainSet)); 
trainSetNew(:,TrLabel90==class)=trainSet(:,TrLabel90==class);
[ DictMat4 , EncoderMat4 ] = TrainDPL(trainSetNew, trls, DictSize, tau, lambda, gamma);

%% classification 
Y=[];coef_norm=[];
 for jj=1 :size(testSet,2) 
TestSet1 = testSet(:,jj);
coef = SimplexRepresentation_acc(trainSet, TestSet1 );
Y=[Y coef];
coef_norm1=[];
for ci = 1:max(TrLabel90)  
coef_c   =  coef(TrLabel90==ci);
%     Dc       =  D(:,Dlabels==ci);
n = norm(coef_c, 2);
coef_norm1=[coef_norm1 n];
end
[dd,class]=max(coef_norm1);
coef_norm=[coef_norm;coef_norm1];
switch class
    case {1}
[ PredictLabel1 Error] = ClassificationDPL( TestSet1, DictMat1, EncoderMat1, DictSize);
    case {2}
[ PredictLabel1 Error] = ClassificationDPL( TestSet1, DictMat2, EncoderMat2, DictSize);
     case {3}
[ PredictLabel1 Error] = ClassificationDPL( TestSet1, DictMat3, EncoderMat3, DictSize);
    case {4}
[ PredictLabel1 Error] = ClassificationDPL( TestSet1, DictMat4, EncoderMat4, DictSize);
end
PredictLabel(1,jj)=PredictLabel1; 
end
Time1 = toc
cornum      =   sum(PredictLabel==ttls);
Rec         =   [cornum/length(ttls)]


