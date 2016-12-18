package.path = 'lib/torch/SCR/' .. "?.lua;" .. package.path

local t = require 'torch'
local matio = require 'matio'
local scr = require 'SCR'

local cmd = t.CmdLine()
cmd:option('-dataFile', '', 'training data file')
cmd:option('-validFile', '', 'validation data file')
cmd:option('-outputFile', '', 'output file')
cmd:option('-maskFile', '', 'mask')
cmd:option('-gpu', '', 'which gpu?')

cmdParams = cmd:parse(arg)

local trainingDataFile = cmdParams.dataFile
local compositionFile = cmdParams.outputFile
local validDataFile = cmdParams.validFile
local maskFile = cmdParams.maskFile
local gpuIdx = cmdParams.gpu

t.setdefaulttensortype('torch.FloatTensor')

print('Loading training data from: ' .. trainingDataFile)

local trainStats = {
   XXt = matio.load(trainingDataFile, 'FFt');
   XYt = matio.load(trainingDataFile, 'DFt');
   YYt = matio.load(trainingDataFile, 'DDt');
}

local featDim = trainStats.XXt:size(1)
local nParams = trainStats.YYt:size(1)

print('Loading validation data from: ' .. validDataFile)

local valStats = {
   XXt = matio.load(validDataFile, 'FFt');
   XYt = matio.load(validDataFile, 'DFt');
   YYt = matio.load(validDataFile, 'DDt');
}

local trainSettings = {
   maxIts = 2500;
   nExamples = matio.load(validDataFile, 'nExamples')[1][1];
   nExamplesTr = matio.load(trainingDataFile, 'nExamples')[1][1]
}

local initComposit = {
   S_1 = matio.load(maskFile, 'S1');
   M_1 = matio.load(maskFile, 'S1m');

   S_2 = matio.load(maskFile, 'S2');
   M_2 = matio.load(maskFile, 'S2m');

   S_3 = t.eye(98, 392);
}

local finalComposit = scr.SolveSparseCompGDScatterSim(trainStats, valStats, trainSettings, initComposit, gpuIdx)


print('Saving composition to disk at ' .. compositionFile)
matio.save(compositionFile, {S1=finalComposit.S_1; S2=finalComposit.S_2; S3=finalComposit.S_3})
