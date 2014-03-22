k = 23;

trainData.numClusters = k;

[trainData.centers, trainData.assignments] = vl_kmeans(double(trainData.concatMatrix),trainData.numClusters,'Initialization', 'plusplus','Algorithm','Elkan');
 
numFeatureVectors = size(trainData.featureVectors);

centerSize = size(trainData.centers);

trainData.featureDescriptors = cell(1,500);
trainData.bagOfWordsHists = zeros(k,500);




for i=1:numFeatureVectors(2)
    trainData.featureDescriptors{i} = zeros(1,featureVectorSize(2));

    for j=1:(featureVectorSize(2))
        trainData.featureDescriptors{i}(1,j) = 1;

        for z=1:(centerSize(2))
           if (vl_alldist2(double(trainData.featureVectors{i}(:,j)),double(trainData.centers(:,z))) < vl_alldist2(double(trainData.featureVectors{i}(:,j)),double(trainData.centers(:,trainData.featureDescriptors{i}(1,j)))))
               trainData.featureDescriptors{i}(1,j) = z;
           end
        end

        trainData.bagOfWordsHists(trainData.featureDescriptors{i}(1,j),i) = trainData.bagOfWordsHists(trainData.featureDescriptors{i}(1,j),i) + 1;
    end
end

trainData.bagOfWordsHists = normc(trainData.bagOfWordsHists);


svmData.class1LabelVector = zeros(1,500);
svmData.class2LabelVector = zeros(1,500);
svmData.class3LabelVector = zeros(1,500);
svmData.class4LabelVector = zeros(1,500);
svmData.class5LabelVector = zeros(1,500);
    

for j=1:500
    if (trainData.labelID{j} == '1')
        svmData.class1LabelVector(1,j) = 1;
    else 
        svmData.class1LabelVector(1,j) = -1;
    end
end

for j=1:500
    if (trainData.labelID{j} == '2')
        svmData.class2LabelVector(1,j) = 1;
    else 
        svmData.class2LabelVector(1,j) = -1;
    end
end

for j=1:500
    if (trainData.labelID{j} == '3')
        svmData.class3LabelVector(1,j) = 1;
    else 
        svmData.class3LabelVector(1,j) = -1;
    end
end

for j=1:500
    if (trainData.labelID{j} == '4')
        svmData.class4LabelVector(1,j) = 1;
    else 
        svmData.class4LabelVector(1,j) = -1;
    end 
end

for j=1:500
    if (trainData.labelID{j} == '5')
        svmData.class5LabelVector(1,j) = 1;
    else 
        svmData.class5LabelVector(1,j) = -1;
    end
end

svmData.W = cell(5,1);
svmData.B = cell(5,1);

[svmData.W{1},svmData.B{1}] = vl_svmtrain(double(trainData.bagOfWordsHists),double(svmData.class1LabelVector),0.0002);
[svmData.W{2},svmData.B{2}] = vl_svmtrain(double(trainData.bagOfWordsHists),double(svmData.class1LabelVector),0.0002);
[svmData.W{3},svmData.B{3}] = vl_svmtrain(double(trainData.bagOfWordsHists),double(svmData.class1LabelVector),0.0002);
[svmData.W{4},svmData.B{4}] = vl_svmtrain(double(trainData.bagOfWordsHists),double(svmData.class1LabelVector),0.0002);
[svmData.W{5},svmData.B{5}] = vl_svmtrain(double(trainData.bagOfWordsHists),double(svmData.class1LabelVector),0.0002);



