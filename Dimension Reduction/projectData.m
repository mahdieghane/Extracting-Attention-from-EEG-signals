
load('D:\darsi\data\SBJ01\SBJ01\S01\Train\trainData');
x=zeros(1600,350);
%feature_extraction(based on Davoodi thesis)
for j=1:1600
    for i=1:8
        
        x(j,:)=x(j,:)+(1/8)*trainData(i,:,j);
        pgama(i,j)=bandpower(pwelch(trainData(i,:,j)),250,[0.5 4]);
        palpha(i,j)=bandpower(pwelch(trainData(i,:,j)),250,[7 14]);
        ptheta(i,j)=bandpower(pwelch(trainData(i,:,j)),250,[4 7]);
        pbeta(i,j)=bandpower(pwelch(trainData(i,:,j)),250,[14 30]);
        xhat(i,j)=mean(trainData(i,:,j));
        variance(i,j)=std(trainData(i,:,j));
        skev(i,j)=(1/350)*sum(trainData(i,:,j)-mean(trainData(i,:,j)))^3/(((1/350)*sum(trainData(i,:,j)-mean(trainData(i,:,j))))^1.5);
        kurtosis(i,j)=(sum(trainData(i,:,j)-mean(trainData(i,:,j)))^4)/(349*(std(trainData(i,:,j))^2));
    end
    
    [ap3(j),dp3(j)]=max(x(j,125:263)-mean(x(j,1:50)));
    %dp3(j)=find(x(j,500:1050)==ap3);
    [ap2(j),dp2(j)]=max(x(j,88:dp3(j)+124)-mean(x(j,1:50)));
    %dp2(j)=find(x(j,350*0.35:(dp3(j)+200)*0.35)==ap2);
    [an2(j),dn2(j)]=min(x(j,dp2(j)+87:dp3(j)+175)-mean(x(j,1:50)));
    %dn2(j)=find(x(j,0.35*(dp2(j)+200):(dp3(j)+200)*0.35)==an2);
    [ap4(j),dp4(j)]= max(x(j,dp3(j)+124:300)-mean(x(j,1:50)));
    %dp4(j)=find(x(j,0.35*(dp3(j)+200):0.35*1200)==ap4);
    [an4(j),dn4(j)]=min(x(j,dp4(j)+dp3(j)+123:300)-mean(x(j,1:50)));
    ap32(j)=ap3(j)-ap2(j);
    dp32(j)=dp3(j)-dp2(j);
    ap3n2(j)=ap3(j)-an2(j);
    dp3n2(j)=dp3(j)-dn2(j);
    ap2n2(j)=ap2(j)-an2(j);
    dp2n2(j)=dp2(j)-dn2(j);
    ap34(j)=ap3(j)-ap4(j);
    dp34(j)=dp3(j)-dp4(j);
    ap4n4(j)=ap4(j)-an4(j);
    dp4n4(j)=dp4(j)-dn4(j);
    ap3n4(j)=ap3(j)-an4(j);
    dp3n4(j)=dp3(j)-dn4(j);
    lar(j)=ap3(j)/dp3(j);
    alar(j)=abs(lar(j));
    ap(j)=0.5*(sum(x(j,125:263))+abs(sum(x(j,125:263))));
    an(j)=0.5*(sum(x(j,125:263))-abs(sum(x(j,125:263))));
    apn(j)=ap(j)+an(j);
    anar(j)=abs(an(j));
    atar(j)=abs(apn(j));
    apnn(j)=ap(j)+abs(an(j));
    aass(j)=1/139*(sum(abs(x(j,125:263)-x(j,126:264))));
    [an3(j),dn3(j)]=min(x(j,125:263)-mean(x(j,1:50)));
    pp(j)=ap3(j)-an3(j);
    tpp(j)=dp3(j)-dn3(j);
    spp(j)=pp(j)/tpp(j);
    n=find(x(j,dn3(j)+125:dp3(j)+125));
    nzc(j)=length(n);
    dzc(j)=nzc(j)/tpp(j);
    nsa(j)=0.5*sum(((x(j,125:263)-x(j,126:264))/abs(x(j,125:263)-x(j,124:262)))+((x(j,125:263)-x(j,124:262))/(abs(x(j,125:263)-x(j,126:264)))));
    
end
Pgama=sum(pgama,2);
    Palpha=sum(palpha,2);
    Ptheta=sum(ptheta,2);
    Pbeta=sum(pbeta,2);

% y=1:350;
% plot(y,x);
% reg=x;
% [b1,a1] = butter(30,0.5);
% reglowbutter = filtfilt(b1,a1,reg);

Data=[ap3',dp3',ap2',dp2',an2',dn2',an4',dn4',ap4',dp4',ap32',ap3n2',ap2n2',dp2n2',ap34',dp34',ap4n4',dp4n4',ap3n4',dp3n4',lar',alar',ap',an',apn',anar',atar',apnn',aass',an3',dn3',pp',tpp',spp',nzc',dzc',nsa',Pgama',Palpha',Ptheta',Pbeta',xhat',variance',skev',kurtosis'];

%[U,S,V]=svds(Data,50);
  %  US=U*S;


