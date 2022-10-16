%clc;

clc;
clear all;

D1=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S01\Train\trainData.mat');
t1=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S01\Train\trainTargets.txt','r');
T1=fscanf(t1,'%f');
D2=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S02\Train\trainData.mat');
t2=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S02\Train\trainTargets.txt','r');
T2=fscanf(t2,'%f');
D3=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S03\Train\trainData.mat');
t3=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S03\Train\trainTargets.txt','r');
T3=fscanf(t3,'%f');
D4=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S04\Train\trainData.mat');
t4=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S04\Train\trainTargets.txt','r');
T4=fscanf(t4,'%f');
D5=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S05\Train\trainData.mat');
t5=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S05\Train\trainTargets.txt','r');
T5=fscanf(t5,'%f');
D6=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S06\Train\trainData.mat');
t6=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S06\Train\trainTargets.txt','r');
T6=fscanf(t6,'%f');
D7=load('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S07\Train\trainData.mat');
t7=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S07\Train\trainTargets.txt','r');
T7=fscanf(t7,'%f');
E1=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S01\Train\trainEvents.txt','r');
e1=fscanf(E1,'%f');
E2=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S02\Train\trainEvents.txt','r');
e2=fscanf(E2,'%f');
E3=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S03\Train\trainEvents.txt','r');
 e3=fscanf(E3,'%f');
 E4=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S04\Train\trainEvents.txt','r');
 e4=fscanf(E4,'%f');
 E5=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S05\Train\trainEvents.txt','r');
 e5=fscanf(E5,'%f');
 E6=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S06\Train\trainEvents.txt','r');
 e6=fscanf(E6,'%f');
 E7=fopen('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S07\Train\trainEvents.txt','r');
 e7=fscanf(E7,'%f');
 
%load('D:\Darsi\Darsi(pervious laptop)\data\SBJ04\SBJ04\S02\Train\trainLabels.mat')
 trainData=cat(3,D1.trainData(:,:,1:896),D2.trainData(:,:,1:896),D3.trainData(:,:,1:896),...
     D4.trainData(:,:,1:896),D5.trainData(:,:,1:896),D6.trainData(:,:,1:896),...
     D7.trainData(:,:,1:896),D1.trainData(:,:,897:1120),D2.trainData(:,:,897:1120),...
     D3.trainData(:,:,897:1120),D4.trainData(:,:,897:1120),D5.trainData(:,:,897:1120),...
     D6.trainData(:,:,897:1120),D7.trainData(:,:,897:1120),D1.trainData(:,:,1121:1600),D2.trainData(:,:,1121:1600),...
     D3.trainData(:,:,1121:1600),D4.trainData(:,:,1121:1600),D5.trainData(:,:,1121:1600),...
     D6.trainData(:,:,1121:1600),D7.trainData(:,:,1121:1600));
 trainTargets=cat(1,T1(1:896,:),T2(1:896,:),T3(1:896,:),T4(1:896,:),T5(1:896,:),...
 T6(1:896,:),T7(1:896,:),T1(897:1120,:),T2(897:1120,:),T3(897:1120,:),T4(897:1120,:),T5(897:1120,:),...
 T6(897:1120,:),T7(897:1120,:),T1(1121:1600,:),T2(1121:1600,:),T3(1121:1600,:),T4(1121:1600,:),...
     T5(1121:1600,:),T6(1121:1600,:),T7(1121:1600,:));
 trainEvents=cat(1,e1(1:896,:),e2(1:896,:),e3(1:896,:),e4(1:896,:),e5(1:896,:),...
 e6(1:896,:),e7(1:896,:),e1(897:1120,:),e2(897:1120,:),e3(897:1120,:),e4(897:1120,:),e5(897:1120,:),...
 e6(897:1120,:),e7(897:1120,:),e1(1121:1600,:),e2(1121:1600,:),e3(1121:1600,:),...
     e4(1121:1600,:),e5(1121:1600,:),e6(1121:1600,:),e7(1121:1600,:));
%trainData=D1.trainData(:,:,1:1600);
%trainTargets=T1(1:1600,:);

train=zeros(250,8,1,1960);
test=zeros(250,8,1,240);

index1=find(trainTargets(1:9520,:)==1);
index0=find(trainTargets(1:9520,:)==0);

train1=trainData(:,:,index1);
train0=trainData(:,:,index0);
%%
ERP=sum(train1,3)/1190;
ERP=sum(ERP,1)/8;
ERP=reshape(ERP,[1,350]);
No_ERP=sum(train0,3)/8330;
No_ERP=sum(No_ERP,1)/8;
No_ERP=reshape(No_ERP,[1,350]);
%%
FS=250;
T=1/FS;
L=350;
for i=1:8
    for j=1:8330
        Y(i,j,:)=fft(train0(i,:,j));
        P21(i,j,:) = abs(Y(i,j,:)/L);
        P11(i,j,:) = P21(i,j,1:L/2+1);
        P11(i,j,2:end-1) = 2*P11(i,j,2:end-1);
        Data_With_LAbel_0(i,(j-1)*350+1:j*350)=train0(i,:,j);
        Band_Fre_0(i,(j-1)*176+1:j*176)=P11(i,j,:);
    end
end



for i=1:8
    for j=1:1190
        Y1(i,j,:)=fft(train1(i,:,j));
        P22(i,j,:) = abs(Y1(i,j,:)/L);
        P12(i,j,:) = P22(i,j,1:L/2+1);
        P12(i,j,2:end-1) = 2*P12(i,j,2:end-1);
        Data_With_LAbel_1(i,(j-1)*350+1:j*350)=train0(i,:,j);
        Band_Fre_1(i,(j-1)*176+1:j*176)=P12(i,j,:);
    end
end

[A,B,C]=csp(Data_With_LAbel_1,Data_With_LAbel_0);
[A1,B1,C1]=csp(Band_Fre_1,Band_Fre_0);

New_Data_1=A'*Data_With_LAbel_1;
New_Data_0=A'*Data_With_LAbel_0;
New_Data_f_1=A1'*Band_Fre_1;
New_Data_f_0=A1'*Band_Fre_0;
for i=1:8
    for j=1:8330
        New_Data0((i-1)*350+1:i*350,j)=New_Data_0(i,(j-1)*350+1:j*350);
        New_Data_f0((i-1)*176+1:i*176,j)=New_Data_f_0(i,(j-1)*176+1:j*176);
    end
end
for i=1:8
    for j=1:1190
        New_Data1((i-1)*350+1:i*350,j)=New_Data_1(i,(j-1)*350+1:j*350);
        New_Data_f1((i-1)*176+1:i*176,j)=New_Data_f_1(i,(j-1)*176+1:j*176);
    end
end

New_Data_f=cat(2, New_Data_f1, New_Data_f0);
New_Data=cat(2,New_Data1, New_Data0);
Label=[ones(1190,1);zeros(8330,1)];
%% Another handy features
%for i=1:8
tData=reshape((sum(trainData,1)/8),[350,11200]);
    for j=1:11200
        P_300(j,1)=max(tData(100:300,j));
        x=zeros(350,1);
        x(:,1)=tData(:,j);
        [Pxx,F] = periodogram(x,hamming(length(x)),length(x),250);
        palpha(j,1)=bandpower(Pxx,F,[7 14],'psd');
        ptheta(j,1)=bandpower(Pxx,F,[4 7],'psd');
        pbeta(j,1)=bandpower(Pxx,F,[14 30],'psd');
        xhat(j,1)=mean(tData(:,j));
        variance(j,1)=std(tData(:,j));
        [ap31(j,1),dp31(j,1)]=max(tData(125:263,j)-mean(tData(1:50,j)));
    [ap21(j,1),dp21(j,1)]=max(tData(88:dp31(j)+124,j)-mean(tData(1:50,j)));
    [an21(j,1),dn21(j,1)]=min(tData(dp21(j)+87:dp31(j)+175,j)-mean(tData(1:50,j)));
    [ap41(j,1),dp41(j,1)]= max(tData(dp31(j)+124:300,j)-mean(tData(1:50,j)));
    [an41(j,1),dn41(j,1)]=min(tData(dp41(j)+dp31(j)+123:300,j)-mean(tData(1:50,j)));
    ap321(j,1)=ap31(j,1)-ap21(j,1);
    dp321(j,1)=dp31(j,1)-dp21(j,1);
    ap3n21(j,1)=ap31(j,1)-an21(j,1);
    dp3n21(j,1)=dp31(j,1)-dn21(j,1);
    ap2n21(j,1)=ap21(j,1)-an21(j,1);
    dp2n21(j,1)=dp21(j,1)-dn21(j,1);
    ap341(j,1)=ap31(j,1)-ap41(j,1);
    dp341(j,1)=dp31(j,1)-dp41(j,1);
    ap4n41(j,1)=ap41(j,1)-an41(j,1);
    dp4n41(j,1)=dp41(j,1)-dn41(j,1);
    ap3n41(j,1)=ap31(j,1)-an41(j,1);
    dp3n41(j,1)=dp31(j,1)-dn41(j,1);
    lar1(j,1)=ap31(j,1)/dp31(j,1);
    alar1(j,1)=abs(lar1(j,1));
    ap1(j,1)=0.5*(sum(tData(125:263,j))+abs(sum(tData(125:263,j))));
    an1(j,1)=0.5*(sum(tData(125:263,j))-abs(sum(tData(125:263,j))));
    apn1(j,1)=ap1(j,1)+an1(j,1);
    anar1(j,1)=abs(an1(j,1));
    atar1(j,1)=abs(apn1(j,1));
    apnn1(j,1)=ap1(j,1)+abs(an1(j,1));
    aass1(j,1)=1/139*(sum(abs(tData(125:263,j)-tData(126:264,j))));
    [an31(j,1),dn31(j,1)]=min(tData(125:263,j)-mean(tData(1:50,j)));
    pp1(j,1)=ap31(j,1)-an31(j,1);
    tpp1(j,1)=dp31(j,1)-dn31(j,1);
    spp1(j,1)=pp1(j,1)/tpp1(j,1);
    n=find(tData(dn31(j,1)+125:dp31(j,1)+125,j));
    nzc1(j,1)=length(n);
    dzc1(j,1)=nzc1(j,1)/tpp1(j,1); 
    %nsa1(j)=0.5*sum(((tData(125:263,j)-tData(126:264,j))/abs(tData(125:263,j)-tData(124:262,j)))+((tData(125:263,j)-tData(124:262,j))/(abs(tData(125:263,j)-tData(126:264,j)))));
        
    end
%end
%% Chioti cFeatures
for i=1:11200
    sampen(j,1)=SampEn_fast(tData(:,j),10,0.5);
    rangeen(j,1)=RangeEn_A(tData(:,j),10,0.5);
    katz(j,1)=Katz_FD(tData(:,j));
    higuchi(j,1)=Higuchi_FD(tData(:,j),5);
    fuzzy(j,1)=hausDim(tData(:,j));
    apen(j,1)=ApEn_fast(tData(:,j),10,0.5);
end
Anotherfeatures=cat(2,sampen,rangeen,katz,higuchi,fuzzy,apen);
Anotherfeatures1=cat(2,palpha,ptheta,pbeta);
%% Putting Features together
Features =cat(2,P_300,palpha,ptheta,pbeta,xhat,variance,ap31,dp31,ap21,dp21,an21,dn21,ap41,dp41,an41,dn41,ap321,dp321,ap3n21,dp3n21,ap2n21,dp2n21,ap341,dp341,ap4n41,dp4n41,ap3n41,dp3n41,lar1,alar1,ap1,an1,apn1,anar1,atar1,apnn1,aass1,an31,dn31,pp1,tpp1,spp1,nzc1,dzc1);
% for i=1:8
%     for j=1:11200
%         feature(45*(i-1)+1:i*45,j)=Features(i,j,:);
%     end
% end

%final_feature=cat(1,New_Data_f,feature(:,1:9520));
%% Imbalancing with repitation
FinalFeatureI_1=zeros(1768,8330);
for i=1:1190
    for j=1:7
    FinalFeatureI_1(:,(i-1)*7+j)=final_feature(:,i);
    end
end
Label_1=cat(1,ones(8330,1),zeros(8330,1));
Data_1=cat(2,FinalFeatureI_1,final_feature(:,1191:9520));
Data_2=cat(2,Data_1',Label_1);
%%
randd=randperm(16660);
Data_3=Data_2(randd,:);
%% SMOTE
idd=randperm(9520);
[final_feature1, final_mark1]=SMOTE(Anotherfeatures1(idd,:),trainTargets(idd,:));
%[final_feature, final_mark]=SMOTE(New_Data',Label);
%%
Table=cat(2,final_feature1,final_mark1);
%% Test
id=randperm(1680);
test=Anotherfeatures1(9521:end,:); 
testLabel=trainTargets(9521:end,:);
%Data1=cat(2,final_feature1,final_mark1);
%Data=cat(2,final_feature,final_mark);

%% Feature Extraction Test
for i=1:8
    for j=1:1680
        Y3(i,j,:)=fft(test(i,:,j));
        P213(i,j,:) = abs(Y(i,j,:)/L);
        P113(i,j,:) = P213(i,j,1:L/2+1);
        P113(i,j,2:end-1) = 2*P113(i,j,2:end-1);
 %       Data_With_LAbel_0(i,(j-1)*350+1:j*350)=train0(i,:,j);
        Band_Fre_test(i,(j-1)*176+1:j*176)=P113(i,j,:);
    end
end

New_Data_test1=A1'*Band_Fre_test;

for i=1:8
    for j=1:1680
        New_Data_test((i-1)*176+1:i*176,j)=New_Data_test1(i,(j-1)*176+1:j*176);
    end
end
%%
predictedlabel=trainedModel10.predictFcn(test);

%% Accuracy
New_Data_test_1=cat(1,feature(:,9521:11200),New_Data_test);
Predicted_Label=trainedModel11.predictFcn(New_Data_test_1');
%%
AC=0;
for i=1:1680
    if predictedlabel(i,:)==testLabel(i,:)
        AC=AC+1;
    end
end
AC/1680
predictedlabel'*testLabel

%% Statistical Analysis
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ01\SBJ01\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ01\SBJ01\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ02\SBJ02\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ02\SBJ02\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ03\SBJ03\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ03\SBJ03\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ04\SBJ04\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ04\SBJ04\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ05\SBJ05\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ05\SBJ05\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ06\SBJ06\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ06\SBJ06\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ09\SBJ09\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ09\SBJ09\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ08\SBJ08\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ08\SBJ08\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ10\SBJ10\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ11\SBJ11\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ11\SBJ11\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ12\SBJ12\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ13\SBJ13\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ13\SBJ13\S02\Train\trainTargets');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ15\SBJ15\S02\Train\trainData');
% load('D:\Darsi\Darsi(pervious laptop)\data\SBJ15\SBJ15\S02\Train\trainTargets');
% 
% [r]=find(trainTargets==1);
% [r1]=find(trainTargets==0);
% 
% data1=trainData(:,:,r);
% data0=trainData(:,:,r1);
% Data1(:,:)=(1/200)*sum(data1,3);
%Data0(:,:)=(1/1400)*sum(data0,3);

% for i=1:8
%     m(i)=max(Data1(i,:));
%     [t(i)]=find(Data1(i,:)==m(i));
% 
% end
% 
% for i=1:1600
%     for j=1:8
%         P_300(i,j)=trainData(j,t(j),i);
%         palpha(i,j)=bandpower(pwelch(trainData(j,:,i)),250,[7 14]);
%         ptheta(i,j)=bandpower(pwelch(trainData(j,:,i)),250,[4 7]);
%         pbeta(i,j)=bandpower(pwelch(trainData(j,:,i)),250,[14 30]);
%         skev(i,j)=(1/350)*sum(trainData(j,:,i)-mean(trainData(j,:,i)))^3/(((1/350)*sum(trainData(j,:,i)-mean(trainData(j,:,i))))^1.5);
% 
%     end
% end
% %clc;
% features(:,:,1)=P_300;
% features(:,:,2)=palpha;
% features(:,:,3)=ptheta;
% features(:,:,4)=pbeta;
% features(:,:,5)=abs(skev);
% 
% tabl = table(trainTargets(1:80,:),features(1:80,1,5),features(1:80,2,5),features(1:80,3,5),...
%     features(1:80,4,5),features(1:80,5,5),features(1:80,6,5),features(1:80,7,5),features(1:80,8,5),...
% 'VariableNames',{'label','features1','features2','features3','features4','features5','features6','features7','features8'});
% featuress = table([1 2 3 4 5 6 7 8]','VariableNames',{'Measurements'});
% rm = fitrm(tabl,'features1-features8~label','WithinDesign',featuress);
% ranovatbl = ttest(rm);
%ranovatbl = anovan(trainTargets(1:80,:),{features(1:80,1,1),features(1:80,2,1),features(1:80,3,1),...
%     features(1:80,4,1),features(1:80,5,1),features(1:80,6,1),features(1:80,7,1),features(1:80,3,2),...
%     features(1:80,4,2),features(1:80,5,2),features(1:80,1,3),features(1:80,2,3),features(1:80,3,3),...
%     features(1:80,4,3),features(1:80,5,3),features(1:80,1,4),features(1:80,2,4),features(1:80,3,4),...
%     features(1:80,4,4),features(1:80,5,4),features(1:80,1,5),features(1:80,2,5),features(1:80,3,5),...
%     features(1:80,4,5),features(1:80,5,5)});
