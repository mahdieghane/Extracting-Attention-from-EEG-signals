for i=1:11200
    for j=1:8
        new_data(8*(i-1)+j,:)= trainData(j,:,i);
    end
    newlabel(8*(i-1)+1:8*i,:)=trainTargets(i,:);
end
newdata=cat(2,new_data,newlabel);
%%
for i=1:8
    for j=1:11200
        Chiotic_features(i,j,:,:,:,:,:,:)=Fuzzy_MI(newdata);
    end
end
%%
for i=1:89600
    Fkatz(i,:)=Katz_FD(new_data(i,:)) ;
end
%%
for i=1:89600
    Fhuchi(i,:)=Higuchi_FD(new_data(i,:),5) ;
end
%%
for i=1:89600
    Frange(i,:)=RangeEn_A(new_data(i,:),5,0.8) ;
end
%%
for i=1:89600
    FAp(i,:)=ApEn_fast(new_data(i,:),5,0.8) ;
    FSa(i,:)=SampEn_fast(new_data(i,:),5,0.8) ;
end
%%
for i=1:89600
    Fcalcp(i,:)=ApEn_fast(new_data(i,:),'exhaustive','true') ;
    Fcalce(i,:)=SampEn_fast(new_data(i,:),'primitive','true') ;
end
%%
t=1;
for i=1:9520
    if trainTargets(i,1)==1
        newtarget(t:t+6,1)=1;
        chiotic_features(t:t+6,1)=Fkatz(1+(i-1)*8:(8*i)-1,:);
        chiotic_features(t:t+6,2)=Fhuchi(1+(i-1)*8:(8*i)-1,:);
        chiotic_features(t:t+6,3)=Frange(1+(i-1)*8:(8*i)-1,:);
        chiotic_features(t:t+6,4)=FAp(1+(i-1)*8:(8*i)-1,:);
        chiotic_features(t:t+6,5)=FSa(1+(i-1)*8:(8*i)-1,:);
                t=t+7;

    elseif trainTargets(i,1)==0
        newtarget(t,1)=0;
        chiotic_features(t,1)=Fkatz(1+(i-1)*8,:);
        chiotic_features(t,2)=Fhuchi(1+(i-1)*8,:);
        chiotic_features(t,3)=Frange(1+(i-1)*8,:);
        chiotic_features(t,4)=FAp(1+(i-1)*8,:);
        chiotic_features(t,5)=FSa(1+(i-1)*8,:);
                t=t+1;

    end
end
%%
Data_4=cat(2,Data_1',chiotic_features,Label_1);
%%
randd=randperm(16660);
Data_5=Data_4(randd,:);

%%

for i=1:1680
    newtest(i,1)=Fkatz(76161+(i-1)*8,:);
    newtest(i,2)=Fhuchi(76161+(i-1)*8,:);
    newtest(i,3)=Frange(76161+(i-1)*8,:);
    newtest(i,4)=FAp(76161+(i-1)*8,:);
    newtest(i,5)=FSa(76161+(i-1)*8,:);
end
%%
new_test=cat(2,New_Data_test_1',newtest);
%%
Predicted_Label=trainedModel124.predictFcn(new_test);
AC=0;
for i=1:1680
    if Predicted_Label(i,:)==testLabel(i,:)
        AC=AC+1;
    end
end
AC/1680