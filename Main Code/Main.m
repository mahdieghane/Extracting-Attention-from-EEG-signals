clc;

dist=zeros(597,1);
index=zeros(597,1);
k=1;
t=1;
%Totaldata=zeros(30,1500,597);
for i=1:length(EEG.epoch)
    Data= EEG.epoch(i);
    Event= cell2mat(Data.eventtype);
    latency= Data.eventlatency;
 %   J=0;
    c=find(Event==251);
    c1=find(Event==252);
    if (size(c)~=0)
            if (cell2mat(latency(c))<=-400)
                J=cat(2,EEG.data(:,round(cell2mat(latency(c))/2)+1700:1500,i-1),EEG.data(:,1:round(cell2mat(latency(c))/2)+1000,i));
            elseif (cell2mat(latency(c))<=1000)
                J=EEG.data(:,round(cell2mat(latency(c))/2)+200:round(cell2mat(latency(c))/2)+1000,i);
            else
                J=cat(2,EEG.data(:,round(cell2mat(latency(c))/2)+200:1500,i),EEG.data(:,1:round(cell2mat(latency(c))/2)-500,i+1));
            end
            e=find(Event==253);
            if size(e)~=0
                Attention(k,1)=cell2mat(latency(e))-cell2mat(latency(c));
                Attention(k,2)=cell2mat(latency(e));
                Attention(k,3)=cell2mat(latency(c));
            end
            if size(e)==0
                Data1=EEG.epoch(i+1);
                Event1= cell2mat(Data1.eventtype);
                latency1=Data1.eventlatency;
                e1=find(Event1==253);
                if size(e1)~=0
                    Attention(k,1)=cell2mat(latency(e1))+1500-cell2mat(latency(c));
                    Attention(k,2)=cell2mat(latency(e1));
                    Attention(k,3)=cell2mat(latency(c));
                end
            end
            k=k+1;
    end
    
    if (size(c1)~=0)
            if (cell2mat(latency(c1))<=-400)
                J=cat(2,EEG.data(:,round(cell2mat(latency(c1))/2)+1700:1500,i-1),EEG.data(:,1:round(cell2mat(latency(c1))/2)+1000,i));
            elseif (cell2mat(latency(c1))<=1000)
                J=EEG.data(:,round(cell2mat(latency(c1))/2)+200:round(cell2mat(latency(c1))/2)+1000,i);
            else
                J=cat(2,EEG.data(:,round(cell2mat(latency(c1))/2)+200:1500,i),EEG.data(:,1:round(cell2mat(latency(c1))/2)-500,i+1));
            end
            e=find(Event==253);
            if size(e)~=0
                Attention(k,1)=cell2mat(latency(e))-cell2mat(latency(c1));
                Attention(k,2)=cell2mat(latency(e));
                Attention(k,3)=cell2mat(latency(c1));
            end
            if size(e)==0
                Data1=EEG.epoch(i+1);
                Event1= cell2mat(Data1.eventtype);
                latency1=Data1.eventlatency;
                e1=find(Event1==253);
                if size(e1)~=0
                    Attention(k,1)=cell2mat(latency(e1))+1500-cell2mat(latency(c1));
                    Attention(k,2)=cell2mat(latency(e1));
                    Attention(k,3)=cell2mat(latency(c1));
                end
            end
            k=k+1;
    end
    Total(k-1,:,:)=J;
end
%%

idx=randperm(size(Attention,1));
TData=Total(idx,:,:);
tTar=Attention(idx,1)/(sum(Attention(idx,1))/length(Attention(idx,1)));
%%
for i=1:251
    if tTar(i,:)<0.8 
        Tar(i,:)=1;
    elseif (tTar(i,:)>=0.8) && (tTar(i,:)<1)
        Tar(i,:)=2;
    else
        Tar(i,:)=3;
    end
    
end
Trtar=categorical(Tar(1:150,:));
Vtar=categorical(Tar(151:200,:));
Ttar=categorical(Tar(201:251,:));

x=zeros(30,801);
for i=1:150
    x(:,:)=TData(i,:,:);
    Train(:,:,1,i)=x(:,:);
end
for i=151:200
    x(:,:)=TData(i,:,:);
    valid(:,:,1,i-150)=x(:,:);
end
for i=201:251
    x(:,:)=TData(i,:,:);
    Test(:,:,1,i-200)=x(:,:);
end

%%
layer=convolution2dLayer([1  65],8,'padding','same');
layer.Name='h';
layer1=batchNormalizationLayer;
layer1.Name='h1';
layer2=convolution2dLayer([8 30],4,'padding','same','NumChannels',2);
%layer2.NumChannels=2;
layer2.Name='h2';
layer3=batchNormalizationLayer;
layer3.Name='h3';
layer4=reluLayer;
layer4.Name='h4';
%layer5= reluLayer;
%layer5.Name='h5';
layer6=averagePooling2dLayer([1 25]);
layer6.Name='h6';

layers = [ ...
     imageInputLayer([30 801 1])
    %%C1 P1
    layer
    layer1
    layer2
    layer3
    layer4
    layer6
    dropoutLayer(0.5)
    %%C2 P2
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
  ];
layer3=averagePooling2dLayer([1 105]);
layer3.Name='h8';
anotherlayers=[imageInputLayer([30 801 1])
convolution2dLayer([30 1],4)
batchNormalizationLayer
reluLayer
layer3
%averagePooling2dLayer([4 1])
dropoutLayer(0.25)
fullyConnectedLayer(3)
softmaxLayer
classificationLayer
];

options = trainingOptions('sgdm',...
'InitialLearnRate',0.0001,...     
    'MaxEpochs',4000,...
'shuffle','every-epoch',...
'ValidationData',{valid,Vtar'},...
'MiniBatchSize',64,...
    'Plots','training-progress');


%net = trainNetwork(Train,Trtar,layers,options);
net1=trainNetwork(Train,Trtar,anotherlayers,options);
%%
inputSize=[30 801 1];
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    imageInputLayer(inputSize,'Name','input')
    
   % sequenceFoldingLayer('Name','fold')
    
    convolution2dLayer([1  65],8,'padding','same','Name','conv1');
    batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    convolution2dLayer([8 30],4,'padding','same','NumChannels',2,'Name','conv2');
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    averagePooling2dLayer([1 25],'Name','Av1')
    
   % sequenceUnfoldingLayer('Name','unfold')
    %flattenLayer('Name','flatten')
    
   % lstmLayer(numHiddenUnits,'Name','lstm')
    
    fullyConnectedLayer(1)
    regressionLayer('Name','regression')];
%%
options = trainingOptions('adam', ...
    'miniBatchSize',20,...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

XTrain=zeros(30,801,1,175);
for i=1:175
XTrain(:,:,1,i)=TData(i,:,:);
end
YTrain=tTar(1:175,:);
%lgraph = layerGraph(layers);
%lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');

net = trainNetwork(XTrain,YTrain,layers,options);
%%

XTest=zeros(30,801,1,76);
for i=1:76
XTest(:,:,1,i)=TData(i+175,:,:);
end
YTest=tTar(176:251,:);
%%
YPredicted = classify(net1,XTest);
%%
predictionError = YTest - double(YPredicted);
%%
numCorrect = sum(abs(predictionError) < 0.2);
numValidationImages = numel(YTest);
accuracy = numCorrect/numValidationImages
squares = predictionError.^2;
rmse = sum(squares)


%%
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end-1:end));

%%

for i=1:251
    for j=1:30
        P_300(i,j)=max(TData(i,j,400:800));
        x=zeros(801,1);
        x(:,1)=TData(i,j,:);
        [Pxx,F] = periodogram(x,rectwin(length(x)),length(x),500);
        palpha(i,j)=bandpower(Pxx,F,[7 14],'psd');
        ptheta(i,j)=bandpower(Pxx,F,[4 7],'psd');
        pbeta(i,j)=bandpower(Pxx,F,[14 30],'psd');
        xhat(i,j)=mean(TData(i,j,:));
        variance(i,j)=std(TData(i,j,:));
        [ap31(i,j),dp31(i,j)]=max(TData(i,j,450:726)-mean(TData(i,j,1:400)));
        [ap21(i,j),dp21(i,j)]=max(TData(i,j,300:dp31(i,j)+354)-mean(TData(i,j,1:400)));
        [an21(i,j),dn21(i,j)]=min(TData(i,j,dp21(i,j)+300:dp31(i,j)+374)-mean(TData(i,j,1:400)));
        [ap41(i,j),dp41(i,j)]= max(TData(i,j,dp31(i,j)+448:800)-mean(TData(i,j,1:300)));
        [an41(i,j),dn41(i,j)]=min(TData(i,j,dp41(i,j)+dp31(i,j)+346:800)-mean(TData(i,j,1:400)));
        ap321(i,j)=ap31(i,j)-ap21(i,j);
        dp321(i,j)=dp31(i,j)-dp21(i,j);
        ap3n21(i,j)=ap31(i,j)-an21(i,j);
        dp3n21(i,j)=dp31(i,j)-dn21(i,j);
        ap2n21(i,j)=ap21(i,j)-an21(i,j);
        dp2n21(i,j)=dp21(i,j)-dn21(i,j);
        ap341(i,j)=ap31(i,j)-ap41(i,j);
        dp341(i,j)=dp31(i,j)-dp41(i,j);
        ap4n41(i,j)=ap41(i,j)-an41(i,j);
        dp4n41(i,j)=dp41(i,j)-dn41(i,j);
        ap3n41(i,j)=ap31(i,j)-an41(i,j);
        dp3n41(i,j)=dp31(i,j)-dn41(i,j);
        lar1(i,j)=ap31(i,j)/dp31(i,j);
        alar1(i,j)=abs(lar1(i,j));
        ap1(i,j)=0.5*(sum(TData(i,j,450:726))+abs(sum(TData(i,j,450:726))));
        an1(i,j)=0.5*(sum(TData(i,j,450:726))-abs(sum(TData(i,j,450:726))));
        apn1(i,j)=ap1(i,j)+an1(i,j);
        anar1(i,j)=abs(an1(i,j));
        atar1(i,j)=abs(apn1(i,j));
        apnn1(i,j)=ap1(i,j)+abs(an1(i,j));
        aass1(i,j)=1/139*(sum(abs(TData(i,j,450:726)-TData(i,j,452:728))));
        [an31(i,j),dn31(i,j)]=min(TData(i,j,450:726)-mean(TData(i,j,1:400)));
        pp1(i,j)=ap31(i,j)-an31(i,j);
        tpp1(i,j)=dp31(i,j)-dn31(i,j);
        spp1(i,j)=pp1(i,j)/tpp1(i,j);
        n=find(TData(i,j,dn31(i,j)+350:dp31(i,j)+350));
        nzc1(i,j)=length(n);
        dzc1(i,j)=nzc1(i)/tpp1(i);
        nsa1(i,j)=0.5*sum(((TData(i,j,450:726)-TData(i,j,452:728))./abs(TData(i,j,450:726)-TData(i,j,448:724)))+((TData(i,j,450:726)-TData(i,j,448:724))./(abs(TData(i,j,450:726)-TData(i,j,452:728)))));
        
    end
end
%%
Features =cat(3,P_300,palpha,ptheta,pbeta,xhat,variance,ap31,dp31,ap21,dp21,an21,dn21,ap41,dp41,an41,dn41,ap321,dp321,ap3n21,dp3n21,ap2n21,dp2n21,ap341,dp341,ap4n41,dp4n41,ap3n41,dp3n41,lar1,alar1,ap1,an1,apn1,anar1,atar1,apnn1,aass1,an31,dn31,pp1,tpp1,spp1,nzc1,dzc1,nsa1);
%%
Trrain=zeros(175,1350);
for i=1:175
    for j=1:30
    Trrain(i,1+(j-1)*45:45*j)=Features(i,j,:);
    end
end

Teest=zeros(76,1350);
for i=1:76
    for j=1:30
    Teest(i,1+(j-1)*45:45*j)=Features(i+175,j,:);
    end
end

tarTrrain=tTar(1:175,:);
tartest=tTar(176:251,:);
%%
mdl=fitlm(Trrain,tarTrrain);
%%
YT=predict(mdl,Teest);
%%
RR=mean((YT-tartest).^2)

R2=1-(RR/sum((tartest-mean(tartest)).^2))
%%
AA=0;
for i=1:76
    if abs(YT(i,:)-tartest(i,:))<0.2
        AA=AA+1;
    end
end
%%
D = 1350;        % full input dimensionalityt
D_eff = 1200;     % number of effective input dimensions
N = 175;        % number of training set examples
N_test = 76;  % number of test set examples
N_plot = 50;    % number of examples to plot

% generate data
w = [randn(D_eff, 1); zeros(D - D_eff, 1)];
% inputs for train & test set
X = Trrain;
X_test = Teest;
% output probabilities and samples for train & test sets
py = 1 ./ (1 + exp(- X * w));
y = tarTrrain;
py_test = 1 ./ (1 + exp(-X_test * w));
y_test = tartest;


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
        mean(2 * (py_VB1 > 0.5) - 1 ~= y));
   % , mean(2 * (py_VB2 > 0.5) - 1 ~= y
    %,VB w/ ARD = %f
    %, mean(2 * (py_test_VB2 > 0.5) - 1 ~= y_test)
fprintf(['test     set 0-1 loss: LDA= %f, VB = %f\n' ...
         '                       VBiter = %f \n'], ...
        mean(y_test_LD ~= y_test), mean(2 * (py_test_VB > 0.5) - 1 ~= y_test), ...
        mean(2 * (py_test_VB1 > 0.5) - 1 ~= y_test));



%     if (cell2mat(Event(1))==253)
%          Data= EEG.epoch(i-1);
%          Event1= Data.eventtype;
% %        if (cell2mat(Event1(end))==252) || (cell2mat(Event1(end))==251)
%          latency1= Data.eventlatency;
%          dist(k,1)=cell2mat(latency(1))-cell2mat(latency1(length(Event1)))+3000;
%          J=cat(2,EEG.data(:,round(cell2mat(latency1(length(Event1)))/2)+301:1500,i-1),EEG.data(:,1:max(0,round(cell2mat(latency1(length(Event1)))/2)),i));
%          Totaldata(:,k,1:size(J,2))=J;
%          index1(t)=i;
%          k=k+1;
%          t=t+1;
% %         elseif (EEG.epoch(i-2).eventtype(end)==251) || (EEG.epoch(i-2).eventtype(end)==252)
% %              Data1= EEG.epoch(i-2);
% %             Event11= Data.eventtype;
% %             latency1= Data.eventlatency;
% %             dist(k,1)=cell2mat(latency(1))-cell2mat(latency1(length(Event1)))+3000;
% %             J=cat(2,EEG.data(:,round(cell2mat(latency1(length(Event1)))/2)+501:1500,i-1),EEG.data(:,1:round(cell2mat(latency(1))/2)+501,i));
% %             Totaldata(:,1:size(J,2),k)=J;
% %             index1(t)=i;
% %             k=k+1;
% %             t=t+1;
% %         end
%     elseif (length(Event)>1) && (cell2mat(Event(2))==253)
%             dist(k,1)=cell2mat(latency(2))-cell2mat(latency(1));
%             
%             J=EEG.data(:,max(1,round(cell2mat(latency(1))/2)+251):min(round(cell2mat(latency(1))/2)+701,1500),i);
%             if round(cell2mat(latency(1))/2)>799
%                 J(:,end+799-round(cell2mat(latency(1))/2):end,i)= EEG.data(:,1:round(cell2mat(latency(1))/2)-799,i+1);
%             elseif round(cell2mat(latency(1))/2)<-250
%                 J(:,500-(-250-round(cell2mat(latency(1))/2)):500,i)=EEG.data(:,round(cell2mat(latency(1))/2)+500:250,i);
%                 J(:,:,:)=J(:,end:-1:1,:);
%             end
%             Totaldata(:,k,1:size(J,2))=J;
%             index(k)=i;
%             k=k+1;
%     elseif (length(Event)>2) && (cell2mat(Event(3))==253)
%             dist(k,1)= cell2mat(latency(3))-cell2mat(latency(2));
%             
%                 J=EEG.data(:,max(0,round(cell2mat(latency(2))/2)+251):min(round(cell2mat(latency(2))/2)+701,1500),i);
%             if round(cell2mat(latency(2))/2)>799
%                 J(:,end+799-round(cell2mat(latency(2))/2):end,i)= EEG.data(:,1:round(cell2mat(latency(2))/2)-799,i+1);
%             elseif round(cell2mat(latency(2))/2)<-250
%                 J(:,500-(-250-round(cell2mat(latency(2))/2)):500,i)=EEG.data(:,1250+round(cell2mat(latency(2))/2):1500,i-1);
%                 J(:,:,:)=J(:,end:1,:);
%             end
%         Totaldata(:,1:size(J,2),k)=J;
%         index(k)=i;
%         k=k+1;
%     elseif (length(Event)>2) && (cell2mat(Event(4))==253)
%         dist(k,1)= cell2mat(latency(4))-cell2mat(latency(3));
%         
%                 J=EEG.data(:,max(0,round(cell2mat(latency(3))/2)+251):min(round(cell2mat(latency(3))/2)+701,1500),i);
%             if round(cell2mat(latency(3))/2)>799
%                 J(:,end+799-round(cell2mat(latency(3))/2):end,i)= EEG.data(:,1:round(cell2mat(latency(3))/2)-799,i+1);
%             elseif round(cell2mat(latency(3))/2)<-250
%                 J(:,500-(-250-round(cell2mat(latency(3))/2)):500,i)=EEG.data(:,1250+round(cell2mat(latency(3))/2):1500,i-1);
%                 J(:,:,:)=J(:,end:1,:);
%             end
%         Totaldata(:,1:size(J,2),k)=J;
%         index(k)=i;
%         k=k+1;
%     end


