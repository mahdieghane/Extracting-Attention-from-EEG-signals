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
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave, rand("state", 0); randn("state", 0); else, rng(0); end
wlims = [-4 4];
for i=1:284
    feature(:,i)=(Feature(:,i)-mean(Feature(:,i)))/var(Feature(:,i));
end

%% settings
D = 284;        % full input dimensionality
D_eff = 10;     % number of effective input dimensions
N = 70;        % number of training set examples
N_test = 10;  % number of test set examples
N_plot = 10;    % number of examples to plot

% generate data
w = [randn(D_eff, 1); zeros(D - D_eff, 1)];
% inputs for train & test set
X = feature(1:70,:);
X_test = feature(71:80,:);
% output probabilities and samples for train & test set
py = 1 ./ (1 + exp(- X * w));
y = 2.^(trainTargets(1:70,:)+1)-3;
py_test = 1 ./ (1 + exp(-X_test * w));
y_test = 2.^(trainTargets(71:80)+1)-3;


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


