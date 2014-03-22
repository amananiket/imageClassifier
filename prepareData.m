clc;
clear;

imageID = 1;

labelsFilePath = 'E:\TagMe!\Train\labels.txt';
basePath = 'E:\TagMe!\Train\Images\';

run('F:\Softwares\vlfeat-0.9.18\toolbox\vl_setup');
imageLabelsFile =  fopen(labelsFilePath,'r');
scannedVector = fscanf(imageLabelsFile,'%c');

trainData.featureVectors = cell(1,500);
trainData.labelID = cell(1,500);


while(1)
    
    initialIndex = (39*(imageID-1)+1);
    finalIndex = (39*(imageID-1)+36);
    labelIndex = 39*(imageID-1)+38;
    
    try
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
        
         

        imageID = imageID + 1;
                 
    catch    
        trainData.concatMatrix = cell2mat(trainData.featureVectors);
       
        break;
    
    end 
        
end