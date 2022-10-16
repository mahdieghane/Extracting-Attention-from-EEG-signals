clc;
load('D:\darsi\data\SBJ01\SBJ01\S01\Train\trainData.mat')
load('D:\darsi\data\SBJ01\SBJ01\S01\Train\trainTargets.mat')
b=1;
wave=zeros(8,351,1600);
for i=1:1600
    for j=1:8
        [ wave(j,:,i)]=wavedec(trainData(j,:,i),3,'db1');
        for k=1:351
            if (8<wave(j,k,i)) && (wave(j,k,i)<12)
                zarayeb(j,b,i)=wave(j,k,i);
                b=b+1;
                
            end
        end
        b=1;
    end
end

