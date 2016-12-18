% SDMComputeBias - This computes the average feature (feature bias)
%
% (C)opyright 2016, Ashton Fagg
function layerBias = SDMComputeBias(imageList, landmarkList, nSynth, model)
images = getListOfFiles(imageList);
landmarks = getListOfFiles(landmarkList);


assert(numel(images) == numel(landmarks));
nFrames = numel(images);


% Fire up worker threads
p = gcp('nocreate');
if (isempty(p))
    p = parpool(4);
end

nChunks = p.NumWorkers;
chunkStats = cell(nChunks, 1);
chunkIndices = generateChunkIndices(nFrames, nChunks);

parfor c = 1:nChunks
    cIdx = find(chunkIndices==c);
    cImages = images(cIdx);
    cLandmarks = landmarks(cIdx);
    chunkStats{c} = doChunk(model, nSynth, cImages, cLandmarks, c);
end


data = struct();
data.nExamples = chunkStats{1}.nExamples;
data.fSum = chunkStats{1}.fSum;

for c = 2:numel(chunkStats)
    data.fSum = data.fSum+chunkStats{c}.fSum;
    data.nExamples = chunkStats{c}.nExamples + data.nExamples;
end

layerBias = (1/data.nExamples) .* data.fSum;
    
end

function list = getListOfFiles(fileList)
    fd = fopen(fileList);
    list = textscan(fd, '%s');
    list = list{1};
    fclose(fd);
end

function currIm = loadIm(imageFile)
    currIm = imread(imageFile);
    if (size(currIm,3) == 3)
        currIm = rgb2gray(currIm);
    end
end

function currPts = loadPts49(ptsFile)
    try
        currPts = load(ptsFile);
    catch
        ff = fopen(ptsFile);
        shape = textscan(ff, '%f %f', 'HeaderLines', 3, 'CollectOutput', ...
                         2);
        fclose(ff);
        currPts = shape{1};
    end
    currPts = currPts([18:60, 62:64, 66:68], :);
end

function currPts = loadPtsAll(ptsFile)
    try
        currPts = load(ptsFile);
    catch
        ff = fopen(ptsFile);
        shape = textscan(ff, '%f %f', 'HeaderLines', 3, 'CollectOutput', ...
                         2);
        fclose(ff);
        currPts = shape{1};
    end
end

function idx = generateChunkIndices(nExamples, nChunks)
    idx = zeros(nExamples, 1);
    for i = 1:numel(idx)
        idx(i) = randi([1 nChunks], 1, 1);
    end
    
end

function frameSum = addFrame(im, pts, model, nSynth, layer)
    frameSum = 0;
    refPts = reshape(model.mu, [], 2);
    nPts = size(refPts, 1);
    
    minX = min(refPts(:,1)) + 1.0;
    minY = min(refPts(:,2)) + 1.0;
    
    refPts = refPts + repmat([minX minY], nPts, 1);
    
    for example = 1:nSynth
        ptsNoise = addSimTNoise(pts, model.mu);
        ptsNoise = SDMApply(im, ptsNoise, model);
        

        [wIm, wPtsIm, wParam, wGt, delta] = doWarp(im, ptsNoise, refPts, pts);
        
        feats = featureExtractDSIFTApprox(model, wIm, wPtsIm);
        frameSum = frameSum+feats(:);
    end
end

function chunkData = doChunk(model, nSynth, images, landmarks, chunkID)
    fprintf('Starting chunk %d\n', chunkID);
    chunkData = struct();
    chunkData.nExamples = 0;
    chunkData.fSum = 0;
    
    nPts = numel(model.mu)/2;
    if (nPts == 49)
        loadPts = @(x) loadPts49(x);
    else
        loadPts = @(x) loadPtsAll(x);
    end
    

    for i = 1:numel(images)
        fprintf('Chunk %d, frame %d of %d -> %d pertubations.\n', chunkID, i, numel(images), nSynth);
        
        
        ff =  addFrame(loadIm(images{i}), loadPts(landmarks{i}), ...
                             model, nSynth, ...
                             numel(model.regressors)+1);
        
        chunkData.nExamples = chunkData.nExamples + nSynth;
        chunkData.fSum = chunkData.fSum + ff;
    end
end


