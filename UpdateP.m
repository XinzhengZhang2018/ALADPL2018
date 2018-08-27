function [ P_Mat ] = UpdateP(  Coef, DataInvMat, P_Mat, DataMat, tau)
% Update P by Eq. (10)

ClassNum = size(Coef,2);
for i=1:ClassNum
     P_Mat{i}           = (tau*Coef{i}*DataMat{i}')*DataInvMat{i};
end

