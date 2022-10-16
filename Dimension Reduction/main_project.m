
clear all
%close all


%data='pima'
nbiter=1;
ratio=0.7;
N=100;
Cvec = logspace(-2,3,N);



options.algo='svmclass';
options.seuildiffsigma=1e-3;
options.seuildiffconstraint=0.1;
options.seuildualitygap=0.01;
options.goldensearch_deltmax=1e-1;
options.numericalprecision=1e-14;
options.stopvariation=0;
options.stopKKT=0;
options.stopdualitygap=1;
options.firstbasevariable='first';
options.nbitermax=500;
options.seuil=0.0;
options.seuilitermax=10;
options.lambdareg = 1e-8;
options.miniter=0;
options.efficientkernel=0;

puiss=0;

lambda = 1e-8; 
span=1;
verbose=1;
x=load('D:\darsi\graduate\term2\bsp\project\codes\main codes\Data.mat');
classcode=[1 -1];

kernelt={'gaussian' 'gaussian' 'poly' 'poly' };
kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20] [0.5 1 2 5 7 10 12 15 17 20] [1 2 3] [1 2 3]};
variablevec={'all' 'single' 'all' 'single'};
%filename=['resultatvariationC-' data '.mat'];
[nbdata,dim]=size(x);
nbtrain=floor(nbdata*ratio)
rand('state',12);

    
    
    

    indice=randperm(nbdata);
    %indapp=indice(1:nbtrain);
    indtest=indice(nbtrain+1:nbdata);
    y1=load('D:\darsi\graduate\term2\bsp\project\codes\main codes\matlab.mat');
    indapp=1200;
    xapp=x.Data(1:indapp,:);
    xtest=x.Data(indapp:end,:);
    yapp=y1.y(1:indapp,:);
    ytest=y1.y(indapp:end,:);

    [xapp,xtest]=normalizemeanstd(xapp,xtest);
    [nbdata,dim]=size(xapp);
    [kernel,kerneloptionvec,optionK.variablecell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
    [K]=mklbuildkernel(xapp,kernel,kerneloptionvec,[],[],optionK);
    option.power=0;
    [K,optionK.weightK]=WeightK(K,option);
    [Kt]=mklbuildkernel(xtest,kernel,kerneloptionvec,xapp,[],optionK);


    
    
    
    
    
    options2=options;
    optionssor=options;
%     first=1;
%     for j=1:length(Cvec); 
%         C=Cvec(j) 
%         if ~first
%             aux  =optionssor.alphainit;
%             aux(find(aux)==Cvec(j-1))=Cvec(j);
%             optionssor.alphainit=aux;
%             optionssor.thetainit=theta;
%         end;
%         tic
%         [w,b,beta,posw,fval(j),storysoren]=mklsvmclassSILP(K,yapp,C,verbose,optionssor);
%         theta=fval(i,j);
%         time(i,j)=toc
%         verbose=1;
%         betavecsoren(j,:)=beta';
%         optionssor.sigmainit=beta';
%         alphainit=zeros(size(yapp));
%         alphainit(posw)=w.*yapp(posw);
%         optionssor.alphainit=alphainit;
%                sumKt=sumKbeta(Kt,beta'.*optionK.weightK);  
%         ypred=sumKt(:,posw)*w +b ; 
%           bcsoren(i,j)=mean(sign(ypred)==ytest);
%         first=0;
%     end    
      first=1;  
    for j=length(Cvec):-1:1
        C=Cvec(j) 
        if ~first
            aux  =options.alphainit;
            aux(find(aux)==Cvec(j+1))=Cvec(j);
            options.alphainit=aux;

        end;
        
        tic
           [beta2,w2,b2,posw,story(j),obj(j)] = mklsvm(K,yapp,C,options,verbose);
           timelasso2(j)=toc
        alphainit=zeros(size(yapp));
        alphainit(posw)=w2.*yapp(posw);
        
        betavec2(j,:)=beta2;
       %sumKt=sumKbeta(Kt,beta2.*optionK.weightK);  
       % ypred=sumKt(:,posw)*w2 +b2 ; 
       %   bc(i,j)=mean(sign(ypred)==ytest);
        
        options.sigmainit=beta2;
        options.alphainit=alphainit;
        first=0;
    end;
    %%

    pp=xapp(posw,:);
    [Kp]=mklbuildkernel(pp,kernel,kerneloptionvec,pp,[],optionK);
    
KK=zeros(length(w2),length(w2));

    for i=1:117
        KK(:,:)=KK(:,:)+(Kp(:,:,i)*beta2(i));
    end
    %it's wrong!
   % A=KK^(0.5);
   %m=find(beta2);
   KK1=zeros(length(beta2),length(beta2));
   h=zeros(1,length(beta2));
   h1=zeros(1,length(beta2));
   %KK1 is phi*phi' and now is wrong!i dont know why!
%    for j=1:length(beta2)
%        for d=1:length(beta2)
%            if d==j
%                t(:,:)=Kp(:,:,d);
%            KK1(d,j)=sum(t*eye(length(w2),length(w2))*ones(length(w2),1)*beta2(d));
%            else
%                h(1,:)=Kp(d,d,:);h1(1,:)=Kp(j,j,:);
%                KK1(d,j)=sqrt(beta2(d)*beta2(j))*sum(((h'*h1)^0.5)*ones(117,1));
%            end
%        end
%    end
%    
%    [V,D]=eig(KK);
%    [V1,D1]=eig(KK1);
  % phi=D\D1;
   
%    for i=1:length(x)
%     phi(:,i)=phi1(x(i));
% end


%k=phi'*phi;
k=KK;

% for i=1:length(beta2)
%     
%     phi1=phi(i,:);
% end
y=yapp(posw);
%y=y*2-1;
alpha=diag(w2)*y;
%Sb=phi*(alpha*alpha')*phi';
Hb=alpha'*alpha;
[N2,N3]=size(pp);
%class1=zeros(N2,1);
%class2=zeros(N2,1);
t=1;
tt=1;
for i=1:length(y)
    if y(i)==1
        class1(t,1)=i;t=t+1;
        
    elseif y(i)==-1
        class2(tt,1)=i;tt=tt+1;
    end
end
t=0;
tt=0;
z1=pp(class1,:);
z2=pp(class2,:);
% for i=1:length(class1)
%    sum1=sum1+(phi1(class1(i))-z1)^2;
% end
% for i=1:length(class2)
%    sum2=sum2+(phi1(class2(i))-z2)^2;
% end
%Sw=sum1+sum2;

for i=1:N2
    for j=1:N2
    if y(i)==1
        if i==j
            l(i,j)=1-(1/(length(class1)));
        elseif y(j)==1
            l(i,j)=-1/(length(class1));
        else l(i,j)=0;
        end
    elseif y(i)==-1
        if i==j
            l(i,j)=1-(1/(length(class2)));
        elseif y(j)==1
            l(i,j)=-1/(length(class2));
        else l(i,j)=0;
        end
    end
    end
end

HW=zeros(N2,N2,N2);
Hw=zeros(N2,N2);

for q=1:N2
    
    HW(:,:,q)=l(q,:)*(l(q,:))';
    Hw=Hw+HW(:,:,q);
    
end


%Hw=sum(l*l');

Kb=k*Hb*k;
Kw=k*Hw*k;

[landa1,landa]=eig(k);
%landa = Hb/Hw;
Kw1=0.95*Kw+(0.05*(trace(Kw)/(N2-2))*eye(N2,N2));
kt=(landa*Kw1)-Kb;
%o=zeros(length(pp),1);
%gama=o/kt;
gama=null(kt);
gama=gama(:,1);

x=xtest;
n=length(x);
%S=zeros(n,n);
P=zeros(n,N2);

 [testkernel]=mklbuildkernel(x,kernel,kerneloptionvec,pp,[],optionK);
 [kernelclass1]=mklbuildkernel(z1,kernel,kerneloptionvec,pp,[],optionK);
 [kernelclass2]=mklbuildkernel(z2,kernel,kerneloptionvec,pp,[],optionK);
 
 M=length(beta2);


    
 for m=1:M
     P(:,:)=P(:,:)+beta2(m)*testkernel(:,:,m);
     
 end
 S=P*diag(gama);

   P1=zeros(length(class1),N2);
   P2=zeros(length(class2),N2);
    
    for m=1:M
        P1(:,:)=P1(:,:)+beta2(m)*kernelclass1(:,:,m);
        
    end
    S1=P1*diag(gama);
    

    for m=1:M
        P2(:,:)=P2(:,:)+beta2(m)*kernelclass2(:,:,m);
        
    end
    S2=P2*diag(gama);
    

F=sum(S)/(gama(:,1)'*k*gama(:,1))^0.5;
F1=F';

FF=sum(S1)/(gama(:,1)'*k*gama(:,1))^0.5;
FF1=FF';

FFF=sum(S2)/(gama(:,1)'*k*gama(:,1))^0.5;
FFF1=FFF';

class1label=mean(FF);
class2label=mean(FFF);

classdata=zeros(n,1);
for a=1:n
    if abs(F1(a)-class1label)>abs(F1(a)-class2label)
        classdata(a)=1;
    elseif (F1(a)-class1label)<abs(F1(a)-class2label)
        classdata(a)=-1;
    end
end
v=zeros(n,1);
for f=1:n
    if classdata(f)==ytest(f)
        v(f)=1;
    else
        v(f)=0;
    end
end
accuracy=sum(v)/n;

plot(Cvec,betavec2,'LineWidth',2);
set(gca,'Xscale','log');
set(gcf,'color','white');
xlabel('C')
ylabel('d_k')
set(gca)


