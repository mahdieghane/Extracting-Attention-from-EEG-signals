clc;

load('D:\darsi\data\SBJ01\SBJ01\S01\Train\trainData.mat')
load('D:\darsi\data\SBJ01\SBJ01\S01\Train\trainTargets.mat')
b=1;
for i=1:1600
    for j=1:8
        [ wave(j,:,i)]=wavedec(trainData(j,:,i),3,'db1');
        for k=1:351
            if (8<wave(j,k,i)) && (wave(j,k,i)<12)
                zarayeb(b,i)=wave(j,k,i);
                b=b+1;
                
            end
        end
       
        for t=1:18
            feature1(i,(j-1)*t+1:j*t)=mean(trainData(j,(t-1)*12+1:t*12,i));
        end
    end
     b=1;
end

Feature=[zarayeb' feature1];
dData=zeros(350,1600);
%dData(:,:)=(1/8)*sum(trainData,1);
dData=Feature';
O=zeros(1600,8);
object=zeros(8,200);
for i=1:8
    object(i,:)=find(trainEvents==i);
    O(object(i,:),i)=1;
end
event=zeros(1600,8);

for i=1:1600
    
end

for i=1:8
    for j=1:1600
       if (O(j,i)==1)&&(trainTargets(j)==1)
           C(j,i)=1;
       elseif (O(j,i)==1)&&(trainTargets(j)==0)
           C(j,i)=0;
       else C(j,i)=2;
       end    

    end
end

c1_1=find(C(:,1)==1);
c2_1=find(C(:,2)==1);
c3_1=find(C(:,3)==1);
c4_1=find(C(:,4)==1);
c5_1=find(C(:,5)==1);
c6_1=find(C(:,6)==1);
c7_1=find(C(:,7)==1);
c8_1=find(C(:,8)==1);

c1_0=find(C(:,1)==0);
c2_0=find(C(:,2)==0);
c3_0=find(C(:,3)==0);
c4_0=find(C(:,4)==0);
c5_0=find(C(:,5)==0);
c6_0=find(C(:,6)==0);
c7_0=find(C(:,7)==0);
c8_0=find(C(:,8)==0);

data_1=[dData(:,c1_1) dData(:,c1_0)];
data_2=[dData(:,c2_1) dData(:,c2_0)];
data_3=[dData(:,c3_1) dData(:,c3_0)];
data_4=[dData(:,c4_1) dData(:,c4_0)];
data_5=[dData(:,c5_1) dData(:,c5_0)];
data_6=[dData(:,c6_1) dData(:,c6_0)];
data_7=[dData(:,c7_1) dData(:,c7_0)];
data_8=[dData(:,c8_1) dData(:,c8_0)];
dd1=[ones(length(c1_1),1);zeros(length(c1_0),1)];
dd2=[ones(length(c2_1),1);zeros(length(c2_0),1)];
dd3=[ones(length(c3_1),1);zeros(length(c3_0),1)];
dd4=[ones(length(c4_1),1);zeros(length(c4_0),1)];
dd5=[ones(length(c5_1),1);zeros(length(c5_0),1)];
dd6=[ones(length(c6_1),1);zeros(length(c6_0),1)];
dd7=[ones(length(c7_1),1);zeros(length(c7_0),1)];
dd8=[ones(length(c8_1),1);zeros(length(c8_0),1)];

Ddata_1=[data_1' dd1];
Ddata_2=[data_2' dd2];
Ddata_3=[data_3' dd3];
Ddata_4=[data_4' dd4];
Ddata_5=[data_5' dd5];
Ddata_6=[data_6' dd6];
Ddata_7=[data_7' dd7];
Ddata_8=[data_8' dd8];


% data1=dData(:,c1);
% data0=dData(:,c0);
% data=[data1';data0'];
% coeff_1=pca(data_1');
% newdata_1=data_1'*coeff_1(:,1:100);
% a_1=[length(c1_1);length(c1_0)];
% [W1,y1]=LDA_ClassIndependent(newdata_1,a_1);
% 
% coeff_2=pca(data_2');
% newdata_2=data_2'*coeff_2(:,1:100);
% a_2=[length(c2_1);length(c2_0)];
% [W2,y2]=LDA_ClassIndependent(newdata_2,a_2);
% 
% coeff_3=pca(data_3');
% newdata_3=data_3'*coeff_3(:,1:100);
% a_3=[length(c3_1);length(c3_0)];
% [W3,y3]=LDA_ClassIndependent(newdata_3,a_3);
% 
% coeff_4=pca(data_4');
% newdata_4=data_4'*coeff_4(:,1:100);
% a_4=[length(c4_1);length(c4_0)];
% [W4,y4]=LDA_ClassIndependent(newdata_4,a_4);
% 
% coeff_5=pca(data_5');
% newdata_5=data_5'*coeff_5(:,1:100);
% a_5=[length(c5_1);length(c5_0)];
% [W5,y5]=LDA_ClassIndependent(newdata_5,a_1);
% 
% coeff_6=pca(data_6');
% newdata_6=data_6'*coeff_6(:,1:100);
% a_6=[length(c6_1);length(c6_0)];
% [W6,y6]=LDA_ClassIndependent(newdata_1,a_1);
% 
% coeff_7=pca(data_7');
% newdata_7=data_7'*coeff_7(:,1:100);
% a_7=[length(c7_1);length(c7_0)];
% [W7,y7]=LDA_ClassIndependent(newdata_7,a_7);
% 
% coeff_8=pca(data_8');
% newdata_8=data_8'*coeff_8(:,1:100);
% a_8=[length(c8_1);length(c8_0)];
% [W8,y8]=LDA_ClassIndependent(newdata_8,a_8);
%% VBR regression

% settings
D = 277;        % full input dimensionality
D_eff = 40;     % number of effective input dimensions
N = 140;        % number of training set examples
N_test = 60;  % number of test set examples
N_plot = 10;    % number of examples to plot

% generate data
w = [randn(D_eff, 1); zeros(D - D_eff, 1)];
% inputs for train & test set
X = [Ddata_8(1:14,1:277);Ddata_8(75:200,1:277)];
X_test = Ddata_8(15:74,1:277);
% output probabilities and samples for train & test set
py = 1 ./ (1 + exp(- X * w));
Targety=[ones(0.7*length(c7_1),1);zeros(round(0.7*length(c7_0)),1)];
y = 2.^(Targety+1)-3;
py_test = 1 ./ (1 + exp(-X_test * w));
Targettest=[ones(0.3*length(c7_1),1);zeros(0.3*length(c7_0),1)];
y_test = 2.^(Targettest+1)-3;


%% estimate coefficients, form predictions for train & test set
% VB, no ARD
fprintf('Variational Bayesian estimation, no ARD\n');
[w_VB, V_VB, invV_VB] = vb_logit_fit(X, y);
py_VB = vb_logit_pred(X, w_VB, V_VB, invV_VB);
py_test_VB = vb_logit_pred(X_test, w_VB, V_VB, invV_VB);
% VB, no hyper-priors, no ARD
fprintf('Variational Bayesian estimation, no ARD, no hyper-priors\n');
[w_VB1, V_VB1, invV_VB1] = vb_logit_fit_iter(X, y);
py_VB1 = vb_logit_pred(X, w_VB1, V_VB1, invV_VB1);
py_test_VB1 = vb_logit_pred(X_test, w_VB1, V_VB1, invV_VB1);
% VB, ARD
fprintf('Variational Bayesian estimation, with ARD\n');
[w_VB2, V_VB2, invV_VB2] = vb_logit_fit_ard(X, y);
py_VB2 = vb_logit_pred(X, w_VB2, V_VB2, invV_VB2);
py_test_VB2 = vb_logit_pred(X_test, w_VB2, V_VB2, invV_VB2);
% Fisher linear discriminant analysis
fprintf('Fisher linear discriminant analysis\n');
y1 = y == 1;
w_LD = (cov(X(y1, :)) + cov(X(~y1, :))) \ ...
       (mean(X(y1, :))' - mean(X(~y1, :))');
c_LD = 0.5 * (mean(X(y1, :)) + mean(X(~y1, :))) * w_LD;
y_LD = 2 * (X * w_LD > c_LD) - 1;
y_test_LD = 2 * (X_test * w_LD > c_LD) - 1;
% output train and test-set error
fprintf(['training set 0-1 loss: LDA    = %f, VB        = %f\n' ...
         '                       VBiter = %f, VB w/ ARD = %f\n'], ...
        mean(y_LD ~= y), mean(2 * (py_VB > 0.5) - 1 ~= y), ...
        mean(2 * (py_VB1 > 0.5) - 1 ~= y), mean(2 * (py_VB2 > 0.5) - 1 ~= y));
fprintf(['test     set 0-1 loss: LDA    = %f, VB        = %f\n' ...
         '                       VBiter = %f, VB w/ ARD = %f\n'], ...
        mean(y_test_LD ~= y_test), mean(2 * (py_test_VB > 0.5) - 1 ~= y_test), ...
        mean(2 * (py_test_VB1 > 0.5) - 1 ~= y_test), ...
        mean(2 * (py_test_VB2 > 0.5) - 1 ~= y_test));


%% plot coefficient estimates
f1 = figure;  hold on;
if exist('wlims', 'var'), xlim(wlims); ylim(wlims); end
% error bars
for i = 1:D
    plot(w(i) * [1 1] - 0.03, w_VB(i) + sqrt(V_VB(i,i)) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.8 0.5 0.5]);
    plot(w(i) * [1 1] - 0.01, w_VB1(i) + sqrt(V_VB1(i,i)) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.8 0.5 0.8]);
    plot(w(i) * [1 1] + 0.01, w_VB2(i) + sqrt(V_VB2(i,i)) * 1.96 * [-1 1], '-', ...
         'LineWidth', 0.25, 'Color', [0.5 0.8 0.5]);
end
% means
h1 = plot(w - 0.03, w_VB, 'o', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'none');
h2 = plot(w - 0.01, w_VB1, 'd', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0.8], 'MarkerEdgeColor', 'none');
h3 = plot(w + 0.01, w_VB2, 's', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'none');
h4 = plot(w + 0.03, w_LD, '+', 'MarkerSize', 3, ...
     'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0 0 0.8], 'LineWidth', 1);
xymin = min([min(xlim) min(ylim)]);  xymax = max([max(xlim) max(ylim)]);
plot([xymin xymax], [xymin xymax], 'k--', 'LineWidth', 0.5);
legend([h1 h2 h3 h4], {'VB', 'VB w/o hyperpriors', 'VB w/ ARD', 'LDA'})
set(gca, 'Box','off', 'PlotBoxAspectRatio', [1 1 1],...
    'TickDir', 'out', 'TickLength', [1 1]*0.02);
xlabel('w');
ylabel('w_{ML}, w_{VB}');


%% plot test set predictions
f2 = figure;  hold on;  xlim([0 1]);  ylim([0 1]);
% misclassification areas
patch([0.5 1 1 0.5], [0 0 0.5 0.5], [0.95 0.8 0.8], 'EdgeColor', 'none');
patch([0 0.5 0.5 0], [0.5 0.5 1 1], [0.95 0.8 0.8], 'EdgeColor', 'none');
% p(y=1) for N_plot samples of test set
h1 = plot(py_test(1:N_plot), py_test_VB(1:N_plot), 'o', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0], 'MarkerEdgeColor', 'none');
h2 = plot(py_test(1:N_plot), py_test_VB1(1:N_plot), 'd', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0.8 0 0.8], 'MarkerEdgeColor', 'none');
h3 = plot(py_test(1:N_plot), py_test_VB2(1:N_plot), 's', 'MarkerSize', 3, ...
     'MarkerFaceColor', [0 0.8 0], 'MarkerEdgeColor', 'none');
plot(xlim, ylim, 'k--', 'LineWidth', 0.5);
legend([h1 h2 h3], {'VB', 'VB w/o hyperpriors', 'VB w/ ARD'})
set(gca, 'Box','off', 'PlotBoxAspectRatio', [1 1 1],...
    'TickDir', 'out', 'TickLength', [1 1]*0.02);
xlabel('p_{true}(y = 1)');
ylabel('p_{VB}(y = 1)');

