%clc;
%clear all;

%load('D:\Darsi\Darsi(pervious laptop)\data\SBJ02\SBJ02\S03\Train\trainData.mat')
%load('D:\Darsi\Darsi(pervious laptop)\data\SBJ02\SBJ02\S03\Train\trainTargets.mat')
%load('D:\Darsi\Darsi(pervious laptop)\data\SBJ02\SBJ02\S03\Train\trainData.mat')
%load('D:\Darsi\Darsi(pervious laptop)\data\SBJ02\SBJ02\S03\Train\trainTargets.mat')

%load('D:\Darsi\Darsi(pervious laptop)\data\SBJ04\SBJ04\S02\Train\trainLabels.mat')

train=zeros(250,8,1,1568);
test=zeros(250,8,1,480);

index1=find(trainTargets(1:1120,:)==1);
index0=find(trainTargets(1:1120,:)==0);

for i=1:length(index1)
    index((i-1)*7+1:i*7)=index1(i);
end
train1=trainData(:,:,index);
train0=trainData(:,:,index0);
Label=cat(1,ones(784,1),zeros(784,1));
Data=cat(3,train1(:,:,197:980),train0(:,:,197:980));
idx=randperm(1568);
Train=Data(:,:,idx);
Ltrain=Label(idx,1);
LabelTrain=categorical(Ltrain);

Label=cat(1,ones(196,1),zeros(196,1));
Data=cat(3,train1(:,:,1:196),train0(:,:,1:196));
idx=randperm(392);
Validation=Data(:,:,idx);
Lvalid=Label(idx,1);
Labelvalid=categorical(Lvalid);

Test=trainData(:,:,1121:1600);
LabelTest=categorical(trainTargets(1121:1600,:));

% Label=cat(1,ones(250,1),zeros(250,1));
% Data=cat(3,train1(:,:,1:250),train0(:,:,1:250));
% idx=randperm(500);
% Test=Data(:,:,idx);
% Ltest=Label(idx,1);
% LabelTest=categorical(Ltest);


for i=1:8
    for j=1:1568
        Train(i,:,j)=(Train(i,:,j)-mean(Train(i,:,j)))/std(Train(i,:,j));
    end
    for j=1:392
        Validation(i,:,j)=(Validation(i,:,j)-mean(Validation(i,:,j)))/std(Validation(i,:,j));
    end
    for j=1:480
        Test(i,:,j)=(Test(i,:,j)-mean(Test(i,:,j)))/std(Test(i,:,j));        
    end
end
Ltest=trainTargets(1121:1600,:);

NewData=cat(3,Train,Validation,Test);
NewLabel=cat(1,Ltrain,Lvalid,Ltest);

for i=1:8
    train(:,i,1,:)=Train(i,50:299,:);
    validation(:,i,1,:)=Validation(i,50:299,:);
    test(:,i,1,:)=Test(i,50:299,:);
end
% LabelTrain=categorical(NewLabel(1:1400));
% Labelvalid=categorical(NewLabel(1401:2100));
% LabelTest=categorical(NewLabel(2101:2800));

% for i=1:2800
%     for j=1:8
%         P_300(i,j)=max(NewData(j,50:299,i));
%         palpha(i,j)=bandpower(pwelch(NewData(j,:,i)),250,[7 14]);
%         ptheta(i,j)=bandpower(pwelch(NewData(j,:,i)),250,[4 7]);
%         pbeta(i,j)=bandpower(pwelch(NewData(j,:,i)),250,[14 30]);
%         skev(i,j)=(1/350)*sum(NewData(j,:,i)-mean(NewData(j,:,i)))^3/(((1/350)*sum(NewData(j,:,i)-mean(NewData(j,:,i))))^1.5);
% 
%     end
% end
%%
layer=convolution2dLayer([65  1],8,'padding',[32 0]);
layer.Name='h';
layer1=batchNormalizationLayer;
layer1.Name='h1';
layer2=convolution2dLayer([1 8],16,'padding',[0 0],'NumChannels',2);
%layer2.NumChannels=2;
layer2.Name='h2';
layer3=batchNormalizationLayer;
layer3.Name='h3';
layer4= reluLayer;
layer4.Name='h4';
layer5= reluLayer;
layer5.Name='h5';
layers = [ ...
     imageInputLayer([250 8 1])
    %%C1 P1
    layer
    layer1
    layer2
    layer3
    layer4
    averagePooling2dLayer([4 1])
    dropoutLayer(0.5)
    %%C2 P2
%     convolution2dLayer([5 5],10,'Padding',[0 5])
%     batchNormalizationLayer
%     layer5
%     averagePooling2dLayer([5 1],'Padding','same')
%     dropoutLayer(0.5);

    %%C3 P3
    %convolution2dLayer(1,16,'Padding','same')
    %batchNormalizationLayer
    %reluLayer
    %averagepooling2dLayer(6,'Padding','same')
    %%C4 P4
    %convolution2dLayer(1,8,'Padding','same')
    %batchNormalizationLayer
    %reluLayer
    %averagePooling2dLayer(6,'Padding','same')
    %% FC1 FC2
    %fullyConnectedLayer(200,'Name','h3')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
  
   
  ];
layer3=averagePooling2dLayer([4 1]);
layer3.Name='h8';
anotherlayers=[imageInputLayer([250 8 1])
convolution2dLayer([1 8],20)
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
'InitialLearnRate',0.0001,...     
    'MaxEpochs',1000,...
'shuffle','every-epoch',...
'ValidationData',{validation,Labelvalid'},...
'MiniBatchSize',64,...
    'Plots','training-progress');
 %   'LearnRateDropFactor',0.1,...
  %  'LearnRateDropPeriod',9,... 
 % 'LearnRateSchedule','piecewise',...

net = trainNetwork(train,LabelTrain,layers,options);
net1=trainNetwork(train,LabelTrain,anotherlayers,options);
%%
%ffeatures=cat(2,P_300,palpha,pbeta,ptheta,abs(skev));

%%
layer3 = 'h4';
layer4='h8';
%featuresTest = activations(net,test,layer,'OutputAs','rows');
for i=1:1960
    x=zeros(250,8,1);
    x1=NewData(:,50:299,i);
    x(:,:,1)=x1';
    featuresTrain = activations(net,x,layer3,'OutputAs','rows');
    anotherfeaturesTrain = activations(net1,x,layer4,'OutputAs','rows');

    Features(:,i)=cat(2,featuresTrain,anotherfeaturesTrain);
   % f1(:,:)=featuresTrain(:,1,:);
   % f2(:,:)=featuresTrain(:,2,:);
    %features(:,i,:)=cat(1,f1,f2); 
end
%Feature={features(:,:,1),features(:,:,2),features(:,:,3),...
%    features(:,:,4),features(:,:,5),features(:,:,6),features(:,:,7),features(:,:,8)};
%%
layer3 = 'h4';
for i=1:480
    x=zeros(250,8,1);
    x1=NewData(:,50:299,i+1960);
    x(:,:,1)=x1';
    featuresTrain = activations(net,x,layer3,'OutputAs','rows');
    anotherfeaturesTrain = activations(net1,x,layer4,'OutputAs','rows');

    Features1(:,i)=cat(2,featuresTrain,anotherfeaturesTrain);
   % f1(:,:)=featuresTrain(:,1,:);
   % f2(:,:)=featuresTrain(:,2,:);
    %features(:,i,:)=cat(1,f1,f2); 
end
%%
[idx,weights] = relieff(Features',NewLabel(1:1960),10);

%%
sfeatures(:,:)=Features(idx,:);
sfeatures1(:,:)=Features1(idx,:);
finalfeatures(:,:)=sfeatures(1:3000,:);
finalfeaturestest(:,:)=sfeatures1(1:3000,:);


%%
Ftfeature=ffeatures(1:2300,:)';
for i=1:2800
ffeatures(i,:)=(ffeatures(i,:)-min(ffeatures(i,:)))/2*(max(ffeatures(i,:))-min(ffeatures(i,:)));
end
finalFtrain=cat(1,Features,ffeatures(1:2300,:)');
finalFtest=cat(1,Features1,ffeatures(2301:2800,:)');


%%
%validationFrequency=floor(numel(2038/100));

numFeatures = 1984;
numHiddenUnits = 200;
numClasses = 2;


layers1 = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    
    softmaxLayer
    classificationLayer];
options1 = trainingOptions('adam', ...
    'MaxEpochs',80, ...
    'GradientThreshold',2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%%
%Label={trainlabel,trainlabel,trainlabel,trainlabel,trainlabel,trainlabel,trainlabel,trainlabel};
net1 = trainNetwork(Features,trainlabel,layers1,options1);

%%
for i=1:300
    x=zeros(250,8,1);
    x1=trainData(:,50:299,i+900);
    x(:,:,1)=x1';
    featurestest = activations(net,x,layer3,'OutputAs','rows');
    testFeatures(:,i)=featuresTrain; 
end
%%
Y=classify(net1,testFeatures);
y=double(string(Y));
accuracy=0;
for i=1:300
if (y(i)==label_test(1,i)) && (label_test(1,i)==1)
accuracy=accuracy+1;
end
end



