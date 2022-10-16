I=imread('C:\Users\ASUS-PLUS\Desktop\chewy-kZRX29xJq6Y-unsplash.jpg');
I1=imresize(I,[224 224]);
net=vgg16();
Layer='pool2';
feature=activations(net,I1,Layer);
for i=1:128
Feature(:,:,i)=((feature(:,:,i)-min(feature(:,:,i)))./(max(feature(:,:,i))-min(feature(:,:,i))))*255;
end

for j=1:8
    for i=1:16
        if i==1
            X=Feature(:,:,(j-1)*16+1);
        else
            X=cat(2,X,Feature(:,:,(j-1)*16+i));
        end
    end
    if j==1
        Y=X;
    else
        Y=cat(1,Y,X);
    end
end
imshow(Y);
