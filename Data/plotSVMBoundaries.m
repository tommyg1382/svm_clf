function [h] = plotSVMBoundaries(labels, features, svmmodel)
%Plot the decision boundaries and data points and SVM model

% Parameters
% training: traning data
% label_train: class lables correspond to training data
% svmmodel: SVM classifier trained using fitcsvm function

% Output
% h: figure handle

% Total number of classes
classes = unique(labels);
nclass =  length(classes);
 
% Set the feature range for ploting
max_x = ceil(max(features(:, 1)));
min_x = floor(min(features(:, 1)));
max_y = ceil(max(features(:, 2)));
min_y = floor(min(features(:, 2)));

xrange = [min_x max_x];
yrange = [min_y max_y];

% step size for how finely you want to visualize the decision boundary.
inc = 0.005;

% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x);
xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.

% distance measure evaluations for each (x,y) pair.
% dist_mat = pdist2(xy, sample_mean);
pred_label = svmpredict(ones(size(xy,1),1), xy, svmmodel, '-q');

% reshape the idx (which contains the class label) into an image.
decisionmap = reshape(pred_label, image_size);

h = figure;
 
%show the image, give each coordinate a color according to its class label
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
 
% colormap for the classes:
cmap = [1 0.8 0.8; 0.9 0.9 1; 0.95 1 0.95;];
colormap (cmap(1:nclass, :));
colorbar;

% plot the class training data.
samplescolor = {'r', 'b', 'g'};
classStr = cell(nclass, 1);
for i = 1:nclass
    scatter(features(labels == classes(i),1),features(labels == classes(i),2), 36, samplescolor{i}, 'filled');
    classStr{i} = sprintf('class %d', classes(i));
end
legend(classStr);

% label the axes.
xlabel('feature 1');
ylabel('feature 2');
end
