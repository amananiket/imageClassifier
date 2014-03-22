clc;
clear;

imageID = 1;

labelsFilePath = 'C:\Users\aman.aniket\Desktop 3\TagMe!\Train\labels.txt';
basePath = 'C:\Users\aman.aniket\Desktop 3\TagMe!\Train\Images\';

run('F:\Softwares\vlfeat-0.9.18\toolbox\vl_setup');
imageLabelsFile =  fopen(labelsFilePath,'r');
scannedVector = fscanf(imageLabelsFile,'%c');

k = floor(sqrt(250));
trainData.featureVectors = cell(1,500);
%trainData.svmMatrix = cell(128,500);
trainData.labelID = cell(1,500);
trainData.numClusters = k;

while(1)
    
    initialIndex = (39*(imageID-1)+1);
    finalIndex = (39*(imageID-1)+36);
    labelIndex = 39*(imageID-1)+38;
    
    
        im = imread(strcat(basePath,scannedVector(initialIndex:finalIndex)));
        fileLabel = scannedVector(labelIndex);
        I = single(rgb2gray(im));

        binSize = 4 ;
        magnif = 3 ;
        Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;

        [f, d] = vl_dsift(Is,'size', binSize,'step',12) ;
        
        
%         [f,d] = vl_sift(I);   
        featureVectorSize = size(trainData.featureVectors{1});
        
        trainData.featureVectors{imageID} = double(d);
        trainData.labelID{imageID} = fileLabel;
        
%         
%         for i=1:featureVectorSize(1)
%             trainData.svmMatrix{i,imageID} = double(d(i,:));
%         end

        imageID = imageID + 1;
                 
        
        trainData.concatMatrix = cell2mat(trainData.featureVectors);
%         trainData.concatSvmMatrix = cell2mat(trainData.svmMatrix);
        
        [trainData.centers, trainData.assignments] = vl_kmeans(double(trainData.concatMatrix),trainData.numClusters,'Initialization', 'plusplus','Algorithm','Elkan');
 
        numFeatureVectors = size(trainData.featureVectors);
        
        centerSize = size(trainData.centers);
        
        trainData.featureDescriptors = cell(1,500);
        trainData.bagOfWordsHists = cell(1,500);
        
        
        
        
        for i=1:numFeatureVectors(2)
            trainData.featureDescriptors{i} = zeros(1,featureVectorSize(2));
            trainData.bagOfWordsHists{i} = zeros(1,centerSize(2));
            
            for j=1:(featureVectorSize(2))
                trainData.featureDescriptors{i}(1,j) = 1;
                
                for z=1:(centerSize(2))
                   if (vl_alldist2(double(trainData.featureVectors{i}(:,j)),double(trainData.centers(:,z))) < vl_alldist2(double(trainData.featureVectors{i}(:,j)),double(trainData.centers(:,trainData.featureDescriptors{i}(1,j)))))
                       trainData.featureDescriptors{i}(1,j) = z;
                   end
                end
                
                trainData.bagOfWordsHists{i}(1,trainData.featureDescriptors{i}(1,j)) = trainData.bagOfWordsHists{i}(1,trainData.featureDescriptors{i}(1,j)) + 1;
            end
        end
        
        trainData.bagOfWordsHists = normc(trainData.bagOfWordsHists);
        
      
        break;
    
    
        
end

svmData.class1LabelVector = zeros(1,500);
svmData.class2LabelVector = zeros(500,1);
svmData.class3LabelVector = zeros(500,1);
svmData.class4LabelVector = zeros(500,1);
svmData.class5LabelVector = zeros(500,1);
    

for j=1:500
    if (trainData.labelID{j} == '1')
        svmData.class1LabelVector(j,1) = 1;
    else 
        svmData.class1LabelVector(j,1) = -1;
    end
end

for j=1:500
    if (trainData.labelID{j} == '2')
        svmData.class2LabelVector(j,1) = 1;
    else 
        svmData.class2LabelVector(j,1) = -1;
    end
end

for j=1:500
    if (trainData.labelID{j} == '3')
        svmData.class3LabelVector(j,1) = 1;
    else 
        svmData.class3LabelVector(j,1) = -1;
    end
end

for j=1:500
    if (trainData.labelID{j} == '4')
        svmData.class4LabelVector(j,1) = 1;
    else 
        svmData.class4LabelVector(j,1) = -1;
    end 
end

for j=1:500
    if (trainData.labelID{j} == '5')
        svmData.class5LabelVector(j,1) = 1;
    else 
        svmData.class5LabelVector(j,1) = -1;
    end
end


[W1,B1] = vl_svmtrain(double(trainData.SvmMatrix),double(svmData.class1LabelVector),0.0002);
W1
