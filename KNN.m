close all;
clear all;
clc;
%%
% K Nearest Neighbor

x_sort=sort(data);
k=5;%1%7%3%5%16
for j=1:length(x)
    [C,I]=min(abs(x_sort-x(j)));
    if I>k && I<N-k
        y=x_sort(I-k:I+k);
    elseif I<k
        y=x_sort(1:2*k+1);
    elseif I<N-k
        y=x_sort(N-2*k-1:N);
    end
    v(j)=abs(x(j)-y(k));
    q(j)=k/(N*v(j));
end
figure
plot(x,q)
axis([0 7 0 4])
title(['KNN with  k=' num2str(k)]) 
