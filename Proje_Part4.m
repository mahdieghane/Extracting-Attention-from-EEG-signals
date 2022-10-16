clc;

for i=1:size(stim8hceeg,2)
    rData(i,:,:,:)=cell2mat(stim8hceeg(1,i));
    Trrain(i,:,:,:)=rData(i,:,1:1000,:);
    TTEST(i,:,:,:)=rData(i,:,1001:2500,:);
end
%%
dataTrain(1,:)=Trrain(1,1,:,1);
dataTest(1,:)=TTEST(1,1,:,1);
%%
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
net = trainNetwork(XTrain,YTrain,layers,options);
%%
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));
%%
numTimeStepsTrain= numel(XTrain);
data(:,:)=cat(2,dataTrain,dataTest);
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
YPred = sig*YPred + mu;
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
%%
net = resetState(net);
[net, pre]= predictAndUpdateState(net,XTrain(end));
YPred(:,1) = pre;
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,pre(1,i-1),'ExecutionEnvironment','cpu');
    if (mod(i,200)==2)
    xx=0;
    yy=0;
    rr=0;
    if YPred(:,i)>max(XTrain)
        YPred(:,i)=max(XTrain);
    elseif YPred(:,i)<min(XTrain)
         YPred(:,i)=min(XTrain);
    end
    if i<51
        y=cat(2,YTrain(1,end-200+i:end),YPred(:,2:i));
    else 
        y=YPred(:,i-199:i);
    end
        [cc,rr]=find((XTrain>YPred(:,i)-0.1) & (XTrain<YPred(:,i)+0.1));
    for j=1:length(rr)
        if rr(1,j)>199
        yy(j,1)=xcorr(y,XTrain(1,rr(1,j)-199:rr(1,j)),0);
%         else
%             break
%         end
%         if rr(1,j)>2
%        if abs(XTrain(1,rr(1,j)-1)-YPred(:,i-1))<5
        if rr(1,j)<798
        if j>2
           if yy(j,1)>yy(j-1,1)
               xx=j;
           end
        else xx=1;
        end
        end
        end
        end
%    end
    pre(:,(200*(i-1))+1:200*i)=XTrain(1,rr(1,xx)+1:rr(1,xx)+200);
end
end
YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-YTest).^2))
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)


