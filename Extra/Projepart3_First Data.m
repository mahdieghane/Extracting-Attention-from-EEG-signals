% clc;
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ02\SBJ02\S01\Train\trainData.mat')
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ02\SBJ02\S01\Train\trainTargets.mat')
% b=1;
% wave=zeros(8,351,1600);
% 
% for i=1:1600
%     for j=1:8
%         [ wave(j,:,i)]=wavedec(trainData(j,:,i),3,'db1');
%         for k=1:351
%             if (8<wave(j,k,i)) && (wave(j,k,i)<12)
%                 zarayeb(b,i)=wave(j,k,i);
%                 b=b+1;
%                 
%             end
%         end
%        
%         for t=1:18
%             feature1(i,(j-1)*t+1:j*t)=mean(trainData(j,(t-1)*12+1:t*12,i));
%         end
%     end
%      b=1;
% end
% 
% Feature=[zarayeb' feature1];
% isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
% if isOctave, rand("state", 0); randn("state", 0); else, rng(0); end
% wlims = [-4 4];
% for i=1:284
%     feature(:,i)=(Feature(:,i)-mean(Feature(:,i)))/var(Feature(:,i));
% end
%%
% sfeatures(:,:)=Features(idx,:);
% sfeatures1(:,:)=Features1(idx,:);
% finalfeatures(:,:)=sfeatures(1:3000,:);
% finalfeaturestest(:,:)=sfeatures1(1:3000,:);
%% Subject out
clc;
clear all;
data1=[];
data0=[];
for t=1:15
   if t~=7
    for j=1:7
        S1=string(j);
        if (t<10)
            S=strcat("0",string(t));
        else
            S=string(t);
        end
        targetd=strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainTargets.mat');
        X=load(targetd);
        datad=strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainData.mat');
        load(datad);
        load(targetd);
        [r]=find(X.T==1);
        [r0]=find(X.T==0);

            data1=cat(3,data1,trainData(:,:,r));
            data0=cat(3,data0,trainData(:,:,r0));

    end
   end
end
Testdata=[];
Testtarget=[];
for i=1:7
    ss=string(i);
    load(strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ07\SBJ07\S0',ss,'\Train\trainData.mat'))
    load(strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ07\SBJ07\S0',ss,'\Train\trainTargets.mat'))
    Testdata=cat(3,Testdata,trainData);
    Testtarget=cat(1,Testtarget,T);
end
New_data1=cat(3,data1,data0(:,:,1:7:end));
New_label1=cat(1,ones(19600,1),zeros(19600,1));
randp=randperm(size(New_data1,3));
New_data=New_data1(:,:,randp);
New_label=New_label1(randp,:);
Train=zeros(8,250,1,30800);
Valid=zeros(8,250,1,8400);
for i=1:8
    Train(i,:,1,:)=New_data(i,50:299,1:30800);
    Valid(i,:,1,:)=New_data(i,50:299,30801:39200);
end
LabelTrain=categorical(New_label(1:30800,:));
LabelValid=categorical(New_label(30801:39200,:));
Test=zeros(8,250,1,11200);
for i=1:8
    Test(i,:,1,:)=Testdata(i,50:299,:);
end

layer=convolution2dLayer([1 65],30,'padding','same');
layer.Name='h';
layer1=batchNormalizationLayer;
layer1.Name='h1';
layer2=groupedConvolution2dLayer([8 1],8,2,'Name','h2');
layer3=batchNormalizationLayer;
layer3.Name='h3';
layer4=reluLayer;
layer4.Name='h4';
layer5=averagePooling2dLayer([1 4]);
layer5.Name='h5';
layer6=convolution2dLayer([1 25],20,'Name','h6');
layer7=batchNormalizationLayer;
layer7.Name='h7';
layer8=reluLayer;
layer8.Name='h8';
layer9=averagePooling2dLayer([1 2]);

layers = [ ...
     imageInputLayer([8 250 1])
    %%C1 P1
    layer
    layer1
    layer8
    layer2
    layer3
    layer4
    layer5
    layer6
 %  layer7
 
   layer9
    dropoutLayer(0.5)
    fullyConnectedLayer(300,'Name','h10')
    reluLayer;
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
  ];

validationfre=floor(numel(LabelTrain)/128);
options1 = trainingOptions('adam',...
    'ExecutionEnvironment','parallel', ... 
'InitialLearnRate',0.0001,...     
    'MaxEpochs',50,...
'shuffle','every-epoch',...
'ValidationData',{Valid,LabelValid'},...
'ValidationFrequency',validationfre,...
'MiniBatchSize',64,...
    'Plots','training-progress');
 %   'LearnRateDropFactor',0.1,...
  %  'LearnRateDropPeriod',20,... 
 % 'LearnRateSchedule','piecewise',...
 %%
net = trainNetwork(Train,LabelTrain,layers,options1);
%% Acuuracy
Predicted=classify(net,Test);
%%
ans=(ans+1.5)/3.5;
F_Label=zeros(11200,1);
for i=1:11200
    if ans(i,:)>0.5
        F_Label(i,:)=1;
    end
end
%%
AC=0;
for i=1:11200
    if Testtarget(i,1)==Predicted_Label(i,1)
    AC=AC+1;
    end
end
accuracy=AC/11200
%%
Ans= F_Label'*Testtarget;
%%
Ans=(Testtarget')*(double(Predicted)-1);
%% Feature Extraction From CNN

layer_Feature = 'h';
for i=1:39200
    x=zeros(8,250,1);
     x(:,:,1)=New_data(:,50:299,i);
    features_Train(i,:) = activations(net,x,layer_Feature,'OutputAs');

end
%%
for i=1:11200
    x=zeros(8,250,1);
     x(:,:,1)=Testdata(:,50:299,i);
    features_Test(i,:) = activations(net,x,layer_Feature,'OutputAs','rows');

end

%% SVM with CNN features
index=relieff(features_Train,New_label,10);
%%
CNN_features=features_Train(:,index(1:300));
Table=cat(2,CNN_features,New_label);
trainedModel=fitcsv(Table,'KernelFunction','polynomial','PolynomialOrder',3);
%%
Predicted_Label=trainedModel.predictFcn(features_Test(:,index(1:300)));
%% Trial out
clc;
clear all;

D1=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S01\Train\trainData.mat');
t1=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S01\Train\trainTargets.txt','r');
T1=fscanf(t1,'%f');
D2=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S02\Train\trainData.mat');
t2=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S02\Train\trainTargets.txt','r');
T2=fscanf(t2,'%f');
D3=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S03\Train\trainData.mat');
t3=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S03\Train\trainTargets.txt','r');
T3=fscanf(t3,'%f');
D4=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S04\Train\trainData.mat');
t4=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S04\Train\trainTargets.txt','r');
T4=fscanf(t4,'%f');
D5=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S05\Train\trainData.mat');
t5=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S05\Train\trainTargets.txt','r');
T5=fscanf(t5,'%f');
D6=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S06\Train\trainData.mat');
t6=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S06\Train\trainTargets.txt','r');
T6=fscanf(t6,'%f');
D7=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S07\Train\trainData.mat');
t7=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S07\Train\trainTargets.txt','r');
T7=fscanf(t7,'%f');
E1=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S01\Train\trainEvents.txt','r');
e1=fscanf(E1,'%f');
E2=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S02\Train\trainEvents.txt','r');
e2=fscanf(E2,'%f');
E3=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S03\Train\trainEvents.txt','r');
 e3=fscanf(E3,'%f');
 E4=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S04\Train\trainEvents.txt','r');
 e4=fscanf(E4,'%f');
 E5=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S05\Train\trainEvents.txt','r');
 e5=fscanf(E5,'%f');
 E6=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S06\Train\trainEvents.txt','r');
 e6=fscanf(E6,'%f');
 E7=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S07\Train\trainEvents.txt','r');
 e7=fscanf(E7,'%f');
 %%Train Size=896*7=6272.Valid=224*7=1568. Test=480*7=3360
%load('D:\Darsi\Darsi(pervious laptop)\data\SBJ04\SBJ04\S02\Train\trainLabels.mat')
 trainData=cat(3,D1.trainData(:,:,1:224),D2.trainData(:,:,1:224),D3.trainData(:,:,1:224),...
     D4.trainData(:,:,1:224),D5.trainData(:,:,1:224),D6.trainData(:,:,1:224),...
     D7.trainData(:,:,1:224),D1.trainData(:,:,225:704),D2.trainData(:,:,225:704),...
     D3.trainData(:,:,225:704),D4.trainData(:,:,225:704),D5.trainData(:,:,225:704),...
     D6.trainData(:,:,225:704),D7.trainData(:,:,225:704),D1.trainData(:,:,705:1600),D2.trainData(:,:,705:1600),...
     D3.trainData(:,:,705:1600),D4.trainData(:,:,705:1600),D5.trainData(:,:,705:1600),...
     D6.trainData(:,:,705:1600),D7.trainData(:,:,705:1600));
 trainTargets=cat(1,T1(1:224,:),T2(1:224,:),T3(1:224,:),T4(1:224,:),T5(1:224,:),...
 T6(1:224,:),T7(1:224,:),T1(225:704,:),T2(225:704,:),T3(225:704,:),T4(225:704,:),T5(225:704,:),...
 T6(225:704,:),T7(225:704,:),T1(705:1600,:),T2(705:1600,:),T3(705:1600,:),T4(705:1600,:),...
     T5(705:1600,:),T6(705:1600,:),T7(705:1600,:));
 trainEvents=cat(1,e1(1:224,:),e2(1:224,:),e3(1:224,:),e4(1:224,:),e5(1:224,:),...
 e6(1:224,:),e7(1:224,:),e1(225:704,:),e2(225:704,:),e3(225:704,:),e4(225:704,:),e5(225:704,:),...
 e6(225:704,:),e7(225:704,:),e1(705:1600,:),e2(705:1600,:),e3(705:1600,:),...
     e4(705:1600,:),e5(705:1600,:),e6(705:1600,:),e7(705:1600,:));
%trainData=D1.trainData(:,:,1:1600);
%trainTargets=T1(1:1600,:);

train=zeros(8,250,1,10976);
test=zeros(8,250,1,3360);

TrainV_target=cat(1,trainTargets(1:1568,:),trainTargets(4929:11200,:));
TrainVdata=cat(3,trainData(:,:,1:1568),trainData(:,:,4929:11200));

idx=randperm(7840);
TrainVdata=TrainVdata(:,:,idx);
TrainV_target=TrainV_target(idx,:);

index1=find(TrainV_target==1);
index0=find(TrainV_target==0);

%%

for i=1:length(index1)
    index((i-1)*7+1:i*7)=index1(i);
end

% for i=1:7:length(index0)
%     index(floor(i/7)+1)=index0(i);
% end

train1=TrainVdata(:,:,index);
train0=TrainVdata(:,:,index0);
Label=cat(1,ones(5488,1),zeros(5488,1));
Data=cat(3,train1(:,:,1:5488),train0(:,:,1:5488));
idx=randperm(10976);
Train=Data(:,:,idx);
Ltrain=Label(idx,1);
LabelTrain=categorical(Ltrain);

Label=cat(1,ones(1372,1),zeros(1372,1));
Data=cat(3,train1(:,:,5489:6860),train0(:,:,5489:6860));
idx=randperm(2744);
Validation=Data(:,:,idx);
Lvalid=Label(idx,1);
Labelvalid=categorical(Lvalid);



% Label=cat(1,ones(250,1),zeros(250,1));
% Data=cat(3,train1(:,:,1:250),train0(:,:,1:250));
% idx=randperm(500);
% Test=Data(:,:,idx);
% Ltest=Label(idx,1);
% LabelTest=categorical(Ltest);

Test=trainData(:,:,1569:4928);
LabelTest=categorical(trainTargets(1569:4928,:));
Ltest=trainTargets(1569:4928,:);
for i=1:8
    for j=1:10976
        Train(i,:,j)=(Train(i,:,j)-mean(Train(i,:,j)))/std(Train(i,:,j));
    end
    for j=1:2744
        Validation(i,:,j)=(Validation(i,:,j)-mean(Validation(i,:,j)))/std(Validation(i,:,j));
    end
    for j=1:3360
        Test(i,:,j)=(Test(i,:,j)-mean(Test(i,:,j)))/std(Test(i,:,j));        
    end
end


NewData=cat(3,Train,Validation,Test);
NewLabel=cat(1,Ltrain,Lvalid,Ltest);

for i=1:8
    train(i,:,1,:)=Train(i,50:299,:);
    validation(i,:,1,:)=Validation(i,50:299,:);
    test(i,:,1,:)=Test(i,50:299,:);
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
layer=convolution2dLayer([1 65],30,'padding','same');
layer.Name='h';
layer1=batchNormalizationLayer;
layer1.Name='h1';
layer2=groupedConvolution2dLayer([8 1],8,2,'Name','h2');
layer3=batchNormalizationLayer;
layer3.Name='h3';
layer4=reluLayer;
layer4.Name='h4';
layer5=averagePooling2dLayer([1 4]);
layer5.Name='h5';
layer6=convolution2dLayer([1 25],50,'Name','h6');
layer7=batchNormalizationLayer;
layer7.Name='h7';
layer8=reluLayer;
layer8.Name='h8';
%layer9=averagePooling2dLayer([8 1]);
%layer9.Name='h9';
% layer2=convolution2dLayer([1 8],16,'padding',[0 2],'NumChannels',2);
% %layer2.NumChannels=2;
% layer2.Name='h2';
% layer3=batchNormalizationLayer;
% layer3.Name='h3';
% layer4=reluLayer;
% layer4.Name='h4';
% %layer5= reluLayer;
% %layer5.Name='h5';
% layer6=averagePooling2dLayer([4 1]);
% layer6.Name='h6';

layers = [ ...
     imageInputLayer([8 250 1])
    %%C1 P1
    layer
    layer1
    layer2
    layer3
    layer4
    layer5
    layer6
   layer7
   layer8
   % averagePooling2dLayer([1 8]);

    %layer9
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
    fullyConnectedLayer(500,'Name','h10')
    reluLayer;
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
  
   
  ];
%%
layer3=convolution2dLayer([1 105],10);
layer3.Name='h9';
anotherlayers=[imageInputLayer([8 250 1])
convolution2dLayer([8 1],20,'Name','h10','stride',1,'padding','same')
batchNormalizationLayer
%reshape(h10.output,[20 250 1])
reluLayer
averagePooling2dLayer([4 1])
layer3
batchNormalizationLayer
reluLayer
avaragepooling2dLayer([1 8])

dropoutLayer(0.25)
fullyConnectedLayer(2)
softmaxLayer
classificationLayer
];

%inputSize = net.Layers(1).InputSize;
%augimdsTrain = augmentedImageDatastore(inputSize(1:2),train);
validationfre=floor(numel(LabelTrain)/128);
options1 = trainingOptions('adam',...
    'ExecutionEnvironment','parallel', ... 
'InitialLearnRate',0.001,...     
    'MaxEpochs',50,...
'shuffle','every-epoch',...
'ValidationData',{Valid,LabelValid'},...
'ValidationFrequency',validationfre,...
'MiniBatchSize',64,...
    'Plots','training-progress');
 %   'LearnRateDropFactor',0.1,...
  %  'LearnRateDropPeriod',20,... 
 % 'LearnRateSchedule','piecewise',...
 %% 
%  Layer1=[imageInputLayer([8 250 1])
% convolution2dLayer(3,8,'Name','h30','padding','same')
% batchNormalizationLayer
% reluLayer
% averagePooling2dLayer(2,'Stride',2)
% convolution2dLayer(3,16,'Name','h31','padding','same')
% batchNormalizationLayer
% reluLayer
% convolution2dLayer(3,32,'Name','h32','padding','same')
% batchNormalizationLayer
% reluLayer
% convolution2dLayer(3,32,'Name','33','padding','same')
% batchNormalizationLayer
% reluLayer
% dropoutLayer(0.2)
% fullyConnectedLayer(2)
% softmaxLayer
% classificationLayer
%  ];
 %%
 
net = trainNetwork(Train,LabelTrain,layers,options1);

%%
net1=trainNetwork(Train,LabelTrain,anotherlayers,options1);
%%
%ffeatures=cat(2,P_300,palpha,pbeta,ptheta,abs(skev));

%%
layer10 = 'h10';
layer11='h12';
%featuresTest = activations(net,test,layer,'OutputAs','rows');
for i=1:5961
    x=zeros(8,350,1);
    x1=zeros(350,8);
    x1(:,:)=FinFeature(i,:,:);
    x(:,:,1)=x1';
    featuresTrain = activations(net,x,layer10,'OutputAs','rows');
    anotherfeaturesTrain = activations(net1,x,layer11,'OutputAs','rows');

    Features(:,i)=cat(2,featuresTrain,anotherfeaturesTrain);
   % f1(:,:)=featuresTrain(:,1,:);
   % f2(:,:)=featuresTrain(:,2,:);
    %features(:,i,:)=cat(1,f1,f2); 
end
%Feature={features(:,:,1),features(:,:,2),features(:,:,3),...
%    features(:,:,4),features(:,:,5),features(:,:,6),features(:,:,7),features(:,:,8)};
%%
layer12 = 'h10';
for i=1:2240
    
    x=zeros(8,350,1);
     x(:,:,1)=trainData(:,:,8960+i);
    featuresTrain = activations(net,x,layer12,'OutputAs','rows');
    anotherfeaturesTrain = activations(net1,x,layer11,'OutputAs','rows');

    Features1(:,i)=cat(2,featuresTrain,anotherfeaturesTrain);
    %f1(:,:)=featuresTrain(:,1,:);
   % f2(:,:)=featuresTrain(:,2,:);
    %features(:,i,:)=cat(1,f1,f2); 
end


%%
[idx,weights] = relieff(Features(:,1000:4000)',final_label(1000:4000,:),10);

%%
sfeatures(:,:)=Features(idx,:);
sfeatures1(:,:)=Features1(idx,:);
finalfeatures(:,:)=sfeatures(1:3000,:);
finalfeaturestest(:,:)=sfeatures1(1:3000,:);

%% settings
D = 300;        % full input dimensionality
D_eff = 250;     % number of effective input dimensions
N = 39200;        % number of training set examples
N_test = 11200;  % number of test set examples
N_plot = 50;    % number of examples to plot

% generate data
w = [randn(D_eff, 1); zeros(D - D_eff, 1)];
% inputs for train & test set
X = double(CNN_features);
%X1=sum(NewData(:,:,1:2300),1)/8;
%X=zeros(2300,250);
%X(:,:)=permute(X1(1,50:299,:),[3 2 1]);
%X1_test=sum(NewData(:,:,2301:2800),1)/8;
%X_test=zeros(500,250);
%X_test(:,:)=permute(X1_test(1,50:299,:),[3 2 1]);
X_test = double(features_Test(:,index(:,1:300)));
% output probabilities and samples for train & test sets
py = 1 ./ (1 + exp(- X * w));
y = 2.^(New_label+1)-3;
py_test = 1 ./ (1 + exp(-X_test * w));
y_test = 2.^(Testtarget+1)-3;


%% estimate coefficients, form predictions for train & test set
% VB, no ARD
fprintf('Variational Bayesian estimation, no ARD\n');
[w_VB, V_VB, invV_VB] = vb_logit_fit(X, y);
py_VB = vb_logit_pred(X, w_VB, V_VB, invV_VB);
py_test_VB = vb_logit_pred(X_test, w_VB, V_VB, invV_VB);
% VB, no hyper-priors, no ARD
fprintf('Variational Bayesian estimation, no ARD, no hyper-priors\n');
[w_VB1, V_VB1, invV_VB1] = vb_logit_fit_iter(X, y);
py_VB1 = vb_logit_pred(X, w_VB1, V_VB1, invV_VB1);
py_test_VB1 = vb_logit_pred(X_test, w_VB1, V_VB1, invV_VB1);
%%
% VB, ARD
  fprintf('Variational Bayesian estimation, with ARD\n');
  [w_VB2, V_VB2, invV_VB2] = vb_logit_fit_ard(X, y);
  py_VB2 = vb_logit_pred(X, w_VB2, V_VB2, invV_VB2);
  py_test_VB2 = vb_logit_pred(X_test, w_VB2, V_VB2, invV_VB2);
  %% Fisher linear discriminant analysis
fprintf('Fisher linear discriminant analysis\n');
y1 = y == 1;
w_LD = (cov(X(y1, :)) + cov(X(~y1, :))) \ ...
       (mean(X(y1, :))' - mean(X(~y1, :))');
c_LD = 0.5 * (mean(X(y1, :)) + mean(X(~y1, :))) * w_LD;
y_LD = 2 * (X * w_LD > c_LD) - 1;
y_test_LD = 2 * (X_test * w_LD > c_LD) - 1;
%%
% output train and test-set error
fprintf(['training set 0-1 loss: LDA = %f, VB = %f\n' ...
         '                       VBiter = %f\n'], ...
        mean(y_LD ~= y), mean(2 * (py_VB > 0.5) - 1 ~= y), ...
        mean(2 * (py_VB2 > 0.5) - 1 ~= y));
   % , mean(2 * (py_VB2 > 0.5) - 1 ~= y
    %,VB w/ ARD = %f
    %, mean(2 * (py_test_VB2 > 0.5) - 1 ~= y_test)
fprintf(['test     set 0-1 loss: LDA= %f, VB = %f\n' ...
         '                       VBiter = %f \n'], ...
        mean(y_test_LD ~= y_test), mean(2 * (py_test_VB > 0.5) - 1 ~= y_test), ...
        mean(2 * (py_test_VB2 > 0.5) - 1 ~= y_test));
    %%
    L=zeros(11200,1);
    for i=1:11200
        if py_test_VB2(i,:)>0.5
            L(i,1)=1;
        end
    end
    %%
    L'*Testtarget
    %%
    K=1*(double(Predicted)-1)+1*(py_test_VB2)+0.8*(Predicted_Label);
    final_label=zeros(11200,1);
    for i=1:8:11200
       [M,id]=max(K(i:i+7,1));
       final_label(id+i-1,1)=1;
    end
    %%
    Ans=(0.5*(final_label'+1))*(0.5*(y_test+1));
    %%
    final_label=2.^(final_label+1)-3;
    Ac=0;
    for i=1:11200
        if final_label(i,1)==y_test(i,1)
            Ac=Ac+1;
        end
    end
Ac/11200
%%
A=zeros(3360,1);
for i=1:3360
  %if py_test_VB(i,:)>0.5
    if (y_test_LD(i,:)==y_test(i,:)) && (y_test(i,:)==1)
        A(i,1)=1;
    end
end
sum(A)/420
%ans=A'*(double(LabelTest)-1);
%ans/420
    %%
    Events=trainEvents(7841:11200,:);
    for i=1:8:length(y_test)-7
        if (mean(2 * (py_test_VB1 > 0.5) - 1 ~= y_test)<0.2)
            [Pmax,indic]=max(py_test_VB1(i:i+7,:));
            [Pmax1,indic1]=max(py_test_VB1(i:i+7,:));
        else
            [Pmax,indic]=max(py_test_VB1(i:i+7,:));
            [Pmax1,indic1]=max(py_test_VB1(i:i+7,:));
        end
        event(floor(i/8)+1)=Events(indic+i-1);
        event1(floor(i/8)+1)=Events(indic1+i-1);
        maxiimax(floor(i/8)+1)=Pmax;
        %maxiimax1(floor(i/8)+1)=Pmax1;
    end
   %%
   for i=1:8:length(y_test)-7
     iidx=find(py_test_VB1(i:i+7)>0.5);
      for j=1:length(iidx)
       eventt(floor(i/8)+1,j)=Events(iidx(j)+i-1);
       end   
    end
    

    
   %%
   for i=1:10:length(event)-9
       k=1;
       k1=1;
       k2=1;
       for j=1:8
           flg=find(event(:,i:i+9)==j);
           flg1=find(event1(:,i:i+9)==j);
           flg2=find(eventt(i:i+9,:)==j);
           if length(flg2)>k2
               numevent=j;
               k=length(flg);
               k1=length(flg1);
               k2=length(flg2);
               %sumi= sum(maxiimax(flg+i-1));
           elseif (length(flg2)==k2) && (k2>1)
               if length(flg)>k
                   k=length(flg);
                   numevent=j;
                  % sumi= sum(maxiimax(flg+i-1));
               elseif length(flg1)>k1
                   k1=length(flg1);
                   numevent=j;
                   k=length(flg);
               elseif (sum(maxiimax(flg+i-1))>sum(maxiimax(flg+i-1)))&& (length(flg1)==k1)
                   k1=length(flg1);
                   numevent=j;
                   k=length(flg);
               end
           end
           
       end
       
       seenevent(floor(i/10)+1)=numevent;
       
   end

%% plot coefficient estimates
% f1 = figure;  hold on;
% if exist('wlims', 'var'), xlim(wlims); ylim(wlims); end
% % error bars
% for i = 1:D
%     plot(w(i) * [1 1] - 0.03, w_VB(i) + sqrt(V_VB(i,i)) * 1.96 * [-1 1], '-', ...
%          'LineWidth', 0.25, 'Color', [0.8 0.5 0.5]);
%     plot(w(i) * [1 1] - 0.01, w_VB1(i) + sqrt(V_VB1(i,i)) * 1.96 * [-1 1], '-', ...
%          'LineWidth', 0.25, 'Color', [0.8 0.5 0.8]);
%     plot(w(i) * [1 1] + 0.01, w_VB2(i) + sqrt(V_VB2(i,i)) * 1.96 * [-1 1], '-', ...
%          'LineWidth', 0.25, 'Color', [0.5 0.8 0.5]);
% end
% % means
% h1 = plot(w - 0.03, w_VB, 'o', 'MarkerSize', 3, ...
%      'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'none');
% h2 = plot(w - 0.01, w_VB1, 'd', 'MarkerSize', 3, ...
%      'MarkerFaceColor', [0.8 0 0.8], 'MarkerEdgeColor', 'none');
% h3 = plot(w + 0.01, w_VB2, 's', 'MarkerSize', 3, ...
%      'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'none');
% h4 = plot(w + 0.03, w_LD, '+', 'MarkerSize', 3, ...
%      'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0 0 0.8], 'LineWidth', 1);
% xymin = min([min(xlim) min(ylim)]);  xymax = max([max(xlim) max(ylim)]);
% plot([xymin xymax], [xymin xymax], 'k--', 'LineWidth', 0.5);
% legend([h1 h2 h3 h4], {'VB', 'VB w/o hyperpriors', 'VB w/ ARD', 'LDA'})
% set(gca, 'Box','off', 'PlotBoxAspectRatio', [1 1 1],...
%     'TickDir', 'out', 'TickLength', [1 1]*0.02);
% xlabel('w');
% ylabel('w_{ML}, w_{VB}');
% 
% 
% %% plot test set predictions
% f2 = figure;  hold on;  xlim([0 1]);  ylim([0 1]);
% % misclassification areas
% patch([0.5 1 1 0.5], [0 0 0.5 0.5], [0.95 0.8 0.8], 'EdgeColor', 'none');
% patch([0 0.5 0.5 0], [0.5 0.5 1 1], [0.95 0.8 0.8], 'EdgeColor', 'none');
% % p(y=1) for N_plot samples of test set
% h1 = plot(py_test(1:N_plot), py_test_VB(1:N_plot), 'o', 'MarkerSize', 3, ...
%      'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'none');
% h2 = plot(py_test(1:N_plot), py_test_VB1(1:N_plot), 'd', 'MarkerSize', 3, ...
%      'MarkerFaceColor', [0.8 0 0.8], 'MarkerEdgeColor', 'none');
% h3 = plot(py_test(1:N_plot), py_test_VB2(1:N_plot), 's', 'MarkerSize', 3, ...
%      'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'none');
% plot(xlim, ylim, 'k--', 'LineWidth', 0.5);
% legend([h1 h2 h3], {'VB', 'VB w/o hyperpriors', 'VB w/ ARD'})
% set(gca, 'Box','off', 'PlotBoxAspectRatio', [1 1 1],...
%     'TickDir', 'out', 'TickLength', [1 1]*0.02);
% xlabel('p_{true}(y = 1)');
% ylabel('p_{VB}(y = 1)');
%%
% layers = [ ...
%      imageInputLayer([250 8 1])
%     %C1 P1
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     averagePooling2dLayer(2,'stride',2)
%     %C2 P2
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     averagePooling2dLayer(2,'stride',2)
%     %C3 P3
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     %C4 P4
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer(0.2);
%     % FC1 FC2
%     fullyConnectedLayer(300,'Name','fully1')
%     reluLayer
%    % fullyConnectedLayer(256)
%     fullyConnectedLayer(2)
%     softmaxLayer
%     classificationLayer
% 
%   ];
% 
% options = trainingOptions('sgdm',...
% 'InitialLearnRate',0.001,...     
% 'LearnRateSchedule','piecewise',...
%     'LearnRateDropFactor',0.1,...
%     'LearnRateDropPeriod',9,... 
%     'MaxEpochs',100,...
% 'shuffle','every-epoch',...
% 'ValidationData',{validation,Labelvalid},...
% 'MiniBatchSize',20,...
%     'Plots','training-progress');
% 

%%

Ytestt=classify(net,test);

%%
Ytesttt=double(Ytestt)-1;
%%
AC=0;
for i=1:length(Ytestt)
    if Ltest(i,:)==Ytesttt(i,:)
        AC=AC+1;
    end
end
%%
Predicted=classify(net,Test);
AC=0;
for i=9601:11200
    if Testtarget(i,1)==double(Predicted(i,:))-1
    AC=AC+1;
    end
end
AC/1600