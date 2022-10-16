x=load('C:\Users\Mahgol\Desktop\simplemkl\pima');
[nbdata,dim]=size(x.X);
ratio=0.7;
nbtrain=floor(nbdata*ratio);
rand('state',12);
    
    

    indice=randperm(nbdata);
    indapp=indice(1:nbtrain);
    indtest=indice(nbtrain+1:nbdata);
    xapp=x.X(indapp,:);
    xtest=x.X(indtest,:);
    yapp=x.y(indapp,:);
    ytest=x.y(indtest,:);


svmmodel=fitcsvm(xapp,yapp);
 
sv = svmmodel.SupportVectors;
figure
gscatter(xapp(:,1),xapp(:,2),yapp)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('versicolor','virginica','Support Vector')
hold off

label=predict(svmmodel,xtest);


v=zeros(length(xtest),1);
for f=1:length(xtest)
    if label(f)==ytest(f)        v(f)=1;
    else
        v(f)=0;
    end
end
accuracy1=sum(v)/length(xtest);