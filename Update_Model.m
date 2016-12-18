clear all, close all;

MODEL_PATH = getenv('MODEL_PATH');
COMPOSIT_PATH = getenv('COMPOSIT_PATH');

load(MODEL_PATH);
load(COMPOSIT_PATH);

cp = cell(3,1);
cp{1} = S1;
cp{2} = S2;
cp{3} = S3;
model.regressors{end+1} = cp;



model.nLayers = model.nLayers + 1;
save(MODEL_PATH, 'model');
