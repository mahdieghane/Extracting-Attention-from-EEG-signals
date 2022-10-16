%clc;
sys=armax(dataTrain,500,'burg');
zarayeb=sys.A;
answ=zeros(500,1);
for j=1:500
    for i=1:length(zarayeb)
        answ(j,1)=answ(j,1)+((j.^(i-1))*((-1)*zarayeb(1,i)));
    end
end
%%

plot(1:500,answ(:,1));
%%

[final_features final_mark] = SMOTE(original_features, original_mark)