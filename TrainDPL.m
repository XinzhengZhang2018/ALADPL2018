function [ DictMat , EncoderMat ] = TrainDPL( Data, Label, DictSize, tau, lambda, gamma )
% This is the DPL training function

% Initilize D and P, precompute the inverse matrix used in Eq. (10), update A for one time 
[ DataMat, DictMat, P_Mat, DataInvMat, CoefMat ] = Initilization( Data , Label, DictSize, tau, lambda, gamma );

% Alternatively update P, D and A
for i=1:20
    [ P_Mat ]   = UpdateP(  CoefMat, DataInvMat, P_Mat, DataMat, tau );
    [ DictMat ] = UpdateD(  CoefMat, DataMat, DictMat);
    [ CoefMat ] = UpdateA(  DictMat, DataMat, P_Mat,  tau, DictSize ); 
end

% Reorganize the P matrix to make the classification fast
for i=1:size(P_Mat,2)
    EncoderMat((i-1)*DictSize+1:i*DictSize,:) = P_Mat{i};
end

    
