% Applies regressors
%
% (C)opyright 2016, Ashton Fagg
function [pts,feats] = SDMApply(im, pts, model)

for layer = 1:numel(model.regressors)
    [pts, feats] = SDMApplySingleLayer(im, pts, model, layer);
end

end
