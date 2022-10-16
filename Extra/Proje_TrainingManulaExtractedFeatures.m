clear all
clc;
for t=1:15
    for j=1:7
        S1=string(j);
        if (t<10)
            S=strcat("0",string(t));
        else
            S=string(t);
        end
%         TargetSTR=strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainTargets.txt');
%         T=importdata(TargetSTR);
%         save(strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainTargets.mat'),'T');
%         LabelSTR=strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainLabels.txt');
%         X=importdata(LabelSTR);
%         save(strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainLabels.mat'),'X');
%         TargetSTR=strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainData');
%         Data=load(TargetSTR);
        targetd=strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainTargets.mat');
        X=load(targetd);
        datad=strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainData');
        load(datad);
        [r]=find(X.T==1);
        [r1]=find(X.T==0);
        
        data1=trainData(:,:,r);
        %data0=trainData(:,:,r1);
        
        Data1(:,:)=(1/200)*sum(data1,3);
        %Data0(:,:)=(1/1400)*sum(data0,3);
        
        for i=1:1600
            for b=1:8
                P_300(i,b)=max(trainData(b,50:200,i));
                palpha(i,b)=bandpower(pwelch(trainData(b,:,i)),250,[7 14]);
                ptheta(i,b)=bandpower(pwelch(trainData(b,:,i)),250,[4 7]);
                pbeta(i,b)=bandpower(pwelch(trainData(b,:,i)),250,[14 30]);
                xhat(i,b)=mean(trainData(b,:,i));
                variance(i,b)=std(trainData(b,:,i));
                [ap31(i,b),dp31(i,b)]=max(trainData(b,125:263,i)-mean(trainData(b,1:50,i)));
                [ap21(i,b),dp21(i,b)]=max(trainData(b,88:dp31(i,b)+124,i)-mean(trainData(b,1:50,i)));
                [an21(i,b),dn21(i,b)]=min(trainData(b,dp21(i,b)+87:dp31(i,b)+175,i)-mean(trainData(b,1:50,i)));
                [ap41(i,b),dp41(i,b)]= max(trainData(b,dp31(i,b)+124:300,i)-mean(trainData(b,1:50,i)));
                [an41(i,b),dn41(i,b)]=min(trainData(b,dp41(i,b)+dp31(i,b)+123:300,i)-mean(trainData(b,1:50,i)));
                ap321(i,b)=ap31(i,b)-ap21(i,b);
                dp321(i,b)=dp31(i,b)-dp21(i,b);
                ap3n21(i,b)=ap31(i,b)-an21(i,b);
                dp3n21(i,b)=dp31(i,b)-dn21(i,b);
                ap2n21(i,b)=ap21(i,b)-an21(i,b);
                dp2n21(i,b)=dp21(i,b)-dn21(i,b);
                ap341(i,b)=ap31(i,b)-ap41(i,b);
                dp341(i,b)=dp31(i,b)-dp41(i,b);
                ap4n41(i,b)=ap41(i,b)-an41(i,b);
                dp4n41(i,b)=dp41(i,b)-dn41(i,b);
                ap3n41(i,b)=ap31(i,b)-an41(i,b);
                dp3n41(i,b)=dp31(i,b)-dn41(i,b);
                lar1(i,b)=ap31(i,b)/dp31(i,b);
                alar1(i,b)=abs(lar1(i,b));
                ap1(i,b)=0.5*(sum(trainData(b,125:263,i))+abs(sum(trainData(b,125:263,i))));
                an1(i,b)=0.5*(sum(trainData(b,125:263,i))-abs(sum(trainData(b,125:263,i))));
                apn1(i,b)=ap1(i,b)+an1(i,b);
                anar1(i,b)=abs(an1(i,b));
                atar1(i,b)=abs(apn1(i,b));
                apnn1(i,b)=ap1(i,b)+abs(an1(i,b));
                aass1(i,b)=1/139*(sum(abs(trainData(b,125:263,i)-trainData(b,126:264,i))));
                [an31(i,b),dn31(i,b)]=min(trainData(b,125:263,i)-mean(trainData(b,1:50,i)));
                pp1(i,b)=ap31(i,b)-an31(i,b);
                tpp1(i,b)=dp31(i,b)-dn31(i,b);
                spp1(i,b)=pp1(i,b)/tpp1(i,b);
                n=find(trainData(b,dn31(i,b)+125:dp31(i,b)+125,i));
                nzc1(i,b)=length(n);
                
                dzc1(i,b)=nzc1(i)/tpp1(i);
                nsa1(i,b)=0.5*sum(((trainData(b,125:263,i)-trainData(b,126:264,i))/abs(trainData(b,125:263,i)-trainData(b,124:262,i)))+((trainData(b,125:263,i)-trainData(b,124:262,i))/(abs(trainData(b,125:263,i)-trainData(b,126:264,i)))));
               % skev(i,b)=(1/350)*sum(trainData(b,:,i)-mean(trainData(b,:,i)))^3/(((1/350)*sum(trainData(b,:,i)-mean(trainData(b,:,i))))^1.5);
            end
        end
        Features(:,:,t,j)=cat(2,P_300,palpha,ptheta,pbeta,xhat,variance,ap31,dp31,ap21,dp21,an21,dn21,ap41,dp41,an41,dn41,ap321,dp321,ap3n21,dp3n21,ap2n21,dp2n21,ap341,dp341,ap4n41,dp4n41,ap3n41,dp3n41,lar1,alar1,ap1,an1,apn1,anar1,atar1,apnn1,aass1,an31,dn31,pp1,tpp1,spp1,nzc1,dzc1,nsa1);
    end
end
%%

for a=1:15
    for b=1:7
        S1=string(b);
        if (a<10)
            S=strcat("0",string(a));
        else
            S=string(a);
        end
        targetd=strcat('D:\Darsi\Darsi(pervious laptop)\data\SBJ',S,'\SBJ',S,'\S0',S1,'\Train\trainTargets.mat');
        X=load(targetd);
        %[r]=find(X.T==1);
        %data1=trainData(:,:,r);
        Target(:,a,b)=X.T;
        for c=1:45
            ffeature(:,1:8,c,a,b)=Features(:,8*(c-1)+1:c*8,a,b);
        tabl = table(Target(:,a,b),ffeature(:,1,c,a,b),ffeature(:,2,c,a,b),ffeature(:,3,c,a,b),...
        ffeature(:,4,c,a,b),ffeature(:,5,c,a,b),ffeature(:,6,c,a,b),ffeature(:,7,c,a,b),ffeature(:,8,c,a,b),...
        'VariableNames',{'label','features1','features2','features3','features4','features5','features6','features7','features8'});
        featuress = table([1 2 3 4 5 6 7 8]','VariableNames',{'Measurements'});
        rm = fitrm(tabl,'features1-features8~label','WithinDesign',featuress);
        ranovatbl = ranova(rm);
        result(a,b,c)=ranovatbl.pValue(1);
        end
    end
end
%%
c=1;
p=1;
%while c<=size(result,3)
for c=1:size(result,3)
    k=0;
    for b=1:7
        for a=1:15
        if result(a,b,c)>0.01
            k=k+1;
        end
        end
    end
    
    
    if k<10
        result_f(:,:,c)=result(:,:,c);
        indx(p,1)=c;
        p=p+1;
    end
end
indx=indx-1;
