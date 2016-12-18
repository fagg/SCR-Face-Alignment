% Extracts BA-SIFT features
%
% Copyright 2016, Ashton Fagg

function feats = featureExtractDSIFTApprox(model, im, pts)
patchSize = 16;
binSize = patchSize / 4;
nBins = 8;
nPts = numel(pts)/2;

[R, C] = size(im);

vec = @(x) x(:);

for pt = 1:nPts
    xCenter = pts(pt,1);
    yCenter = pts(pt,2);

    xCenterFloor=floor(xCenter);
    yCenterFloor=floor(yCenter);
    xRatio=xCenter-xCenterFloor;
    yRatio=yCenter-yCenterFloor;

    idx1R = yCenterFloor + [-patchSize/2:patchSize/2-1];
    idx1C = xCenterFloor + [-patchSize/2:patchSize/2-1];

    idx2R = yCenterFloor + [-patchSize/2:patchSize/2-1];
    idx2C = xCenterFloor + 1 + [-patchSize/2:patchSize/2-1];
    
    idx3R = yCenterFloor + 1 + [-patchSize/2:patchSize/2-1];
    idx3C = xCenterFloor + [-patchSize/2:patchSize/2-1];
    
    idx4R = yCenterFloor + 1 + [-patchSize/2:patchSize/2-1];
    idx4C = xCenterFloor + 1 + [-patchSize/2:patchSize/2-1]; 



    patch = single(im(fixBounds(idx1R, R), fixBounds(idx1C, C)))*(1-xRatio)*(1-yRatio);
    patch = patch + single(im(fixBounds(idx2R, R), fixBounds(idx2C, C)))*(xRatio)*(1-yRatio);
    patch = patch + single(im(fixBounds(idx3R, R), fixBounds(idx3C, C)))*(1-xRatio)*(yRatio);
    patch = patch + single(im(fixBounds(idx4R, R), fixBounds(idx4C, C)))*(xRatio)*(yRatio);
    
    Rx = zeros(16, 16);
    Ry = zeros(16, 16);

    Ry(1,:) = (patch(2,:)-patch(1,:));
    Ry(2:end-1,:) = (patch(3:end,:)-patch(1:end-2,:));
    Ry(end,:) = (patch(end,:)-patch(end-1,:));

    Rx(:,1) = (patch(:,2)-patch(:,1));
    Rx(:,2:end-1) = (patch(:,3:end)-patch(:,1:end-2));
    Rx(:,end) = (patch(:,end)-patch(:,end-1));
    
    Rx = Rx(:);
    Ry = Ry(:);
    
    ff = [];
    binPatt = [Rx>0 Ry>0 abs(Rx)>abs(Ry)];
    featCode = 4*binPatt(:,1) + 2*binPatt(:,2) + binPatt(:,3);
    featCode = featCode + 1;
    
    featCode=changem(featCode,[6,5,3,4,7,8,2,1],[1,2,3,4,5,6,7,8]);
    
    histo = cell(8,1);
    [histo{:}] = deal(zeros(256,1));
    
    for p = 1:256
        histo{featCode(p)}(p) = 1;
        if (featCode(p)==8)
            idx = 1;
        else
            idx = featCode(p) + 1;
        end
        histo{idx}(p) = 1;
    end
    
    histo = cellfun(@(grad) reshape(grad, [16 16]), histo, 'uniformoutput', false);
    ff = sign(model.featApprox)*vec(cell2mat(histo));
    
    
    ff = ff / (norm(ff) + eps);
    %    ff(ff>0.2) = 0.2;
    %ff = ff / (norm(ff) + eps);
    feats(:,pt) = ff;
end

end

function idx = fixBounds(x, upper)
    for i = 1:numel(x)
        if (x(i) < 1)
            idx(i) = 1;
        elseif (x(i) > upper)
            idx(i) = upper;
        else
            idx(i) = x(i);
        end
    end
    
end


