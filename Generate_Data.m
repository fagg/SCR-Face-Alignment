clear all, close all;
addpath(genpath('lib/matlab'));


FIT_FILE = getenv('FIT_FILE');
LANDMARK_FILES = getenv('LANDMARK_FILE_LIST');
IMAGE_FILES = getenv('IMAGE_FILE_LIST');
MODEL_PATH = getenv('MODEL_PATH');
MODEL_NAME = getenv('MODEL_NAME');
PDM_PATH = getenv('PDM_PATH');

VAL_FILE = getenv('VAL_FILE');
TRAIN_FILE = getenv('TRAIN_FILE');

VAL_IMAGES = getenv('VAL_IMAGE_LIST');
VAL_LANDMARKS = getenv('VAL_LANDMARK_LIST');

if (exist(MODEL_PATH))
    fprintf('Loading model: %s\n', MODEL_PATH);
    load(MODEL_PATH);
    noiseParams = [];
else
    fprintf('Init new model.\n');
    model = SDMInitModel(MODEL_NAME);
    pdm = load(PDM_PATH);
    model.mu = pdm.mu;
    model.bias = {};
    
    load('featState.mat');
    model.featApprox = R;
    clear R;
end

sIdx = model.nLayers + 1;

fprintf('Computing bias...\n');
model.bias{end+1} = SDMComputeBias(IMAGE_FILES, LANDMARK_FILES, 5, model);

fprintf('Saving model to: %s\n', MODEL_PATH);
save(MODEL_PATH, 'model');

fprintf('Generating training data...\n');
trainingData = SDMAddDataMemoryFrugal(IMAGE_FILES, LANDMARK_FILES, 5, model);

fprintf('Generating validation data...\n');
valData = SDMAddDataMemoryFrugalValidation(VAL_IMAGES, VAL_LANDMARKS, 2, model, 3000);


fprintf('Dumping generated data...\n');
nExamples = trainingData.nExamples;
FFt = trainingData.FFt;
DFt = trainingData.DFt;
DDt = trainingData.DDt;

save(TRAIN_FILE, 'FFt', 'DFt', 'DDt', 'nExamples');

nExamples = valData.nExamples;
FFt = valData.FFt;
DFt = valData.DFt;
DDt = valData.DDt;

save(VAL_FILE, 'FFt', 'DFt', 'DDt', 'nExamples');


                         



