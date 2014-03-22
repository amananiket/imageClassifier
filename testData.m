
run('F:\Softwares\vlfeat-0.9.18\toolbox\vl_setup');

labelsFilePath = 'E:\TagMe!\Test\feature_vectors.txt';
basePath = 'E:\TagMe!\Test\Images\';
outputFile =  fopen('E:\TagMe!\Test\labels.txt','w');


imageLabelsFile =  fopen(labelsFilePath,'r');
scan = fscanf(imageLabelsFile,'%s');
detectedWord = '';

fileNamesArray = cell(3809,1);
bagOfWordsHist = zeros(k,3809);

fileID = 1;

letters = size(scan);

for i=2:letters(2)
   
    if(scan(i) == 'j' && scan(i-1) == '.')
        initialIndex = i-33;
        finalIndex = i;
        fileNamesArray{fileID} = strcat(scan(initialIndex:finalIndex),'pg');
        fileID = fileID + 1;
    end

end
        
for i=1:3809

    im = imread(strcat(basePath,fileNamesArray{i}));
    I = single(rgb2gray(im));
    
    fprintf(outputFile,'%s',fileNamesArray{i});

    
    binSize = 4 ;
    magnif = 3 ;
    Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;

    [f, d] = vl_dsift(Is,'size', binSize,'step',12) ;
    
    s = size(d);
    
    featureDescriptors = zeros(1,s(2));
   

    for j=1:s(2)

        featureDescriptors(1,j) = 1;

        for z=1:k
           if (vl_alldist2(double(d(:,j)),double(trainData.centers(:,z))) < vl_alldist2(double(d(:,j)),double(trainData.centers(:,featureDescriptors(1,j)))))
               featureDescriptors(1,j) = z;
           end
        end

        bagOfWordsHist(featureDescriptors(1,j),i) = bagOfWordsHist(featureDescriptors(1,j),i) + 1;
    end
    
    scoresArray = zeros(5,1);

    label = 0;
    for t=1:5
        scoresArray(t,1) = svmData.W{t}'*bagOfWordsHist(:,i)+ svmData.B{t};
    end
    
    [C,I] = max(scoresArray);
    
    fprintf(outputFile,'%s',' ');
    fprintf(outputFile,'%s\n',int2str(I));
    
    
end

