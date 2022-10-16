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
                J=cat(2,EEG.data(:,round(cell2mat(latency(c))/2)+1700:1500,i-1),EEG.data(:,1:round(cell2mat(latency(c))/2)+800,i));
            else
                J=EEG.data(:,round(cell2mat(latency(c))/2)+200:round(cell2mat(latency(c))/2)+800,i);
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
                J=cat(2,EEG.data(:,round(cell2mat(latency(c1))/2)+1700:1500,i-1),EEG.data(:,1:round(cell2mat(latency(c1))/2)+800,i));
            else
                J=EEG.data(:,round(cell2mat(latency(c1))/2)+200:round(cell2mat(latency(c1))/2)+800,i);
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
Tar=Attention(idx,1);
Trtar=categorical(Tar(1:190,:));
Vtar=categorical(Tar(191:200,:));
tTar=Attention(idx,1);
for i=1:252
    if tTar(i,:)<750 
        Tar(i,:)=1;
    elseif (tTar(i,:)>=750) && (tTar(i,:)<1000)
        Tar(i,:)=2;
    else
        Tar(i,:)=3;
    end
    
end

Trtar=categorical(Tar(1:190,:));
Vtar=categorical(Tar(191:200,:));
Ttar=categorical(Tar(201:252,:));

x=zeros(30,601);
for i=1:190
    x(:,:)=TData(i,:,:);
    Train(:,:,1,i)=x(:,:);
end
for i=191:200
    x(:,:)=TData(i,:,:);
    valid(:,:,1,i-190)=x(:,:);
end
for i=201:252
    x(:,:)=TData(i,:,:);
    Test(:,:,1,i-200)=x(:,:);
end

%%
layer=convolution2dLayer([30  1],12,'padding','same');
layer.Name='h';
layer1=batchNormalizationLayer;
layer1.Name='h1';
layer2=convolution2dLayer([1 75],4,'padding','same','NumChannels',2);
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
     imageInputLayer([30 601 1])
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
anotherlayers=[imageInputLayer([30 601 1])
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
    'MaxEpochs',50,...
'shuffle','every-epoch',...
'ValidationData',{valid,Vtar'},...
'MiniBatchSize',64,...
    'Plots','training-progress');


net = trainNetwork(Train,Trtar',layers,options);
net1=trainNetwork(Train,Trtar',anotherlayers,options);

