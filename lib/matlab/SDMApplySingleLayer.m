% SDMApplySingleLayer - Applies single SCR regressor
%
% Copyright 2016, Ashton Fagg

function [pts, feats] = SDMApplySingleLayer(im, pts, model, layer)
refPts = reshape(model.mu, [numel(model.mu)/2 2]);
nPts = size(refPts, 1);

minX = min(refPts(:,1)) + 1.0;
minY = min(refPts(:,2)) + 1.0;

refPts = refPts + repmat([minX minY], nPts, 1);

A = calcSimT(pts, refPts);
wPts = [pts, ones(numel(model.mu)/2, 1)] * A';
[wIm, wPtsIm, wParam] = doWarp(im, pts, refPts);

feats = featureExtractDSIFTApprox(model, wIm, wPtsIm);
feats = feats(:)-model.bias{layer};


delta = (model.regressors{layer}{3}*(model.regressors{layer}{2}*(model.regressors{layer}{1}* feats)));



pts = [wPts + reshape(delta, [], 2), ones(numel(model.mu)/2, 1)] * invSimT(A)';


end
