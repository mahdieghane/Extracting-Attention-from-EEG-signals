clc;
TRDATA=zeros(2800,11200);
for i=1:11200
    for j=1:8
        TRDATA(i,350*(j-1)+1:350*j)=trainData(j,:,i);
    end
end
[final_features final_mark]=SMOTE(TRDATA,trainTargets);
