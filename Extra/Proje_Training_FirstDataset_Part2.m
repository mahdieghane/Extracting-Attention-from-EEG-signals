clc;
DDD=zeros(2800,8960);
%%
for i=1:8
    DDD(1+(i-1)*350:350*i,:)=trainData(i,:,1:8960);
end
%%

for i=1:8
    DDD1(1+(i-1)*350:350*i,:)=trainData(i,:,8961:11200);
end
%%
DDD2=cat(2,DDD1',trainTargets(8961:11200,:));
%%
acur=0;
for i=1:2240
    if (yfit(i,:)==trainTargets(8960+i,:)) && (yfit(i,:)==1)
        acur=acur+1;
    end
end
acur=acur/2240
%%
[final_features final_label] = SMOTE(DDD',trainTargets(1:8960,:));
%%
for i=1:8
FinFeature(:,:,i)=final_features(:,1+(i-1)*350:i*350);
end
%%
pidx=randperm(5685);
FinFeature=FinFeature(pidx,:,:);
%%
pidx=randperm(5685);
final_features=final_features(pidx,:);
%%
final_label1=final_label(pidx,:);
Train1=zeros(8,350,1,4600);
%%
for i=1:6220
    k=find(pidx==i);
    pidxT(1,i)=k;
end
%%
final_label1=final_label(pidxT,:);
%%
DDD3=cat(2,final_features,final_label1);
%%
LabelTrain=categorical(final_label(1:4600,:));
LabelValid=categorical((final_label(4601:5961,:)));
%%
LabelTest=trainTargets(8961:11200,:);
%%
Train1=zeros(8,350,1,4600);
Validation1=zeros(8,350,1,1361);
%%
Test1=zeros(8,350,1,2240);
%%
for i=1:8
Train1(i,:,1,:)=FinFeature(1:4600,:,i)';
Validation1(i,:,1,:)=FinFeature(4601:5961,:,i)';
Test1(i,:,1,:)=trainData(i,:,8962:11200);
end

%%

layer=convolution2dLayer([1  65],8,'padding',[0 32]);
layer.Name='h';
layer1=batchNormalizationLayer;
layer1.Name='h1';
layer2=groupedConvolution2dLayer([8,1],16,2,'Name','h2');
layer3=batchNormalizationLayer;
layer3.Name='h3';
layer4=reluLayer;
layer4.Name='h4';
layer5=averagePooling2dLayer([1 4]);
layer5.Name='h5';
layer6=groupedConvolution2dLayer([1 17],16,'channel-wise','padding',[0,8],'Name','h6');
layer3=batchNormalizationLayer;
layer3.Name='h7';
layer4=reluLayer;
layer4.Name='h8';
layer5=averagePooling2dLayer([1 8]);
layer5.Name='h9';
layers = [ ...
     imageInputLayer([8 350 1])
    %%C1 P1
    layer
    layer1
    layer2
    layer3
    layer4
    layer6
    dropoutLayer(0.5)
    %%C2 P2
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer 
  ];
%%
layer3=averagePooling2dLayer([1 4]);
layer3.Name='h9';
anotherlayers=[imageInputLayer([8 350 1])
convolution2dLayer([8 1],20)
batchNormalizationLayer
reluLayer
layer3
%averagePooling2dLayer([4 1])
dropoutLayer(0.25)
fullyConnectedLayer(2)
softmaxLayer
classificationLayer
];

%inputSize = net.Layers(1).InputSize;
%augimdsTrain = augmentedImageDatastore(inputSize(1:2),train);
options = trainingOptions('sgdm',...
'InitialLearnRate',0.001,...     
    'MaxEpochs',50,...
'shuffle','every-epoch',...
'ValidationData',{Validation1,LabelValid},...
'MiniBatchSize',64,...
    'Plots','training-progress');
 %   'LearnRateDropFactor',0.1,...
  %  'LearnRateDropPeriod',9,... 
 % 'LearnRateSchedule','piecewise',...

net = trainNetwork(Train1,LabelTrain,layers,options);
net1=trainNetwork(Train1,LabelTrain,anotherlayers,options);
%%
LLabel=classify(net,Test1);
LLabel1=classify(net1,Test1);
%%
ted=zeros(8,350,1,2240);
for i=1:8
    ted(i,:,1,:)=double(LabelTest(i,:,:));
end


%% 
accuracy=0;
for i=1:2240
    if (LabeLL(i,:)==LabelTest(i,:)) && (LabeLL(i,:)==1)
        accuracy=accuracy+1;
    end
end

%%
LL=cat(2,LL1(1,:),LL2(2,:),LL3(3,:),LL4(4,:),LL5(5,:),LL6(6,:),LL7(7,:),LL8(8,:));
FF=cat(2,FF1(1,:,:),FF2(2,:,:),FF3(3,:,:),FF4(4,:,:),FF5(5,:,:),FF6(6,:,:),FF7(7,:,:),FF8(8,:,:));
iidx=find(LL==1);
NewData1(:,:)=FF(1,iidx,:);
%%
for i=1:8
    NewData2(i,:,:)=NewData1(1+(i-1)*5600:i*5600,:);
end