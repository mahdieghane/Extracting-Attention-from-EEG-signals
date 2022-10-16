%% Example of how to use the KFDA class
% The data generated here doesn't have the best shape to demonstrate the
% value of KFDA, but at least you have an example of how to use the class

% First execute the first part of the code to train and test the model,
% then execute the second part to visualize the decision boundary

% Generating fake data as two 2nd order polynomials that cannot be
% separated linearly
x=load('C:\Users\Mahgol\Desktop\simplemkl\pima');
[nbdata,dim]=size(x.X);
ratio=0.7
nbtrain=floor(nbdata*ratio)
rand('state',12);
    
    

    indice=randperm(nbdata);
    indapp=indice(1:nbtrain);
    indtest=indice(nbtrain+1:nbdata);
    xapp=x.X(indapp,:);
    xtest=x.X(indtest,:);
    yapp=x.y(indapp,:);
    ytest=x.y(indtest,:);
    
    t=1;
    tt=1;
    for i=1:length(xapp)
        if yapp(i)==1
            clas1(t)=i;t=t+1;
        elseif yapp(i)==-1
            clas2(tt)=i;tt=tt+1;
        end
    end
cllas1=xapp(clas1,:);
cllas2=xapp(clas2,:);
yclas1=yapp(clas1,:);
yclas2=yapp(clas2,:);
% Concatenate your data into a single matrix (row are observations, columns
% are features or variables)
X = [[cllas1', yclas1']; [cllas2', yclas2']];
Y = cell(length(xapp), 1);
% The class labels must be in a cell
Y(1:length(clas1)) = {'class1'};
Y(length(clas1)+1:length(clas1)+length(clas2)) = {'class2'};

% Creating and training the KFDA object
kfda = KFDA(X, Y, 3);

% Testing the model by generating new data points with the same two
% polynomials
% x1test = linspace(1, 5);
% y1test = (x1test-3).^2 + rand(1, 100)*2;
% x2test = linspace(3, 7);
% y2test = -1.5*(x2test-5).^2 + rand(1, 100)*2 + 8;
% 
% Xtest = [[x1test', y1test']; [x2test', y2test']];
% Ytest = cell(200, 1);
% Ytest(1:100) = {'class1'};
% Ytest(101:200) = {'class2'};

ccllas1=xapp(cclas1,:);
ccllas2=xapp(cclas2,:);
ycclas1=yapp(cclas1,:);
ycclas2=yapp(cclas2,:);
% Concatenate your data into a single matrix (row are observations, columns
% are features or variables)
Xtest = [[ccllas1', ycclas1']; [ccllas2', ycclas2']];
Ytest = cell(length(xapp), 1);
% The class labels must be in a cell
Ytest(1:length(cclas1)) = {'class1'};
Ytest(length(cclas1)+1:length(cclas1)+length(cclas2)) = {'class2'};

% Predicting the label of these new points
results = kfda.predict(Xtest, Ytest);

% The following figure shows the new data points are circles in red the
% ones that have been misclassified
plot(x1test, y1test, 'xg');
hold on
plot(x2test, y2test, 'xb');

for iobs = 1:200
    if results.errors(iobs) == 1
        plot(Xtest(iobs, 1), Xtest(iobs, 2), 'ro')
    end
end

%% Here is a way to get an idea of the decision boundary of this model:
xlin = linspace(0, 8);
ylin = linspace(-1, 11);

ys = repmat(ylin, 100, 1);
Xdb = [repmat(xlin', 100, 1) ys(:)];

% Generating fake labels as we are not interested in the performance but
% just the prediction of the model
nprobes = size(Xdb, 1);
Ydb = cell(nprobes, 1);
Ydb(:) = {'class1'};

resdb = kfda.predict(Xdb, Ydb);

figure()
plot(x1, y1, 'xg');
hold on
plot(x2, y2, 'xb');

for iprobe = 1:nprobes
    if strcmp(resdb.predictedClass{iprobe}, 'class1')
        marker = '.g';
    else
        marker = '.b';
    end
    plot(Xdb(iprobe, 1), Xdb(iprobe, 2), marker)
end

% The color of each dot represents the class predicted at that point
% The 'x's correspond to the original data used to train the model')

%% Other methods:

% Project data on the the 'nDims' dimension
projdata = kfda.project_data(Xtest, 1);

% Find the Symmetrised Kullback - Leibler divergence between classes
kldivergence = kfda.KL_divergence();
