# SCR for Face Alignment

ArXiv paper: https://arxiv.org/abs/1612.05332

This is the reference implementation for Sparse-Compositional Regression based face landmark alignment. This implementation
is not designed to be fast, it is merely for training and basic testing.

A fast implementation (for online fitting), written in C++ will be added soon.

The included model is trained on the 300W dataset using 49 points (no jaw points).

Training code is included in the lib directory. Training should be a fairly straight-forward process. The code is based largely
on my SDM implementation: https://github.com/fagg/SDM-Face-Alignment

The functions for training are:

- SDMInitModel - Allocates a new SDM model
- SDMAddDataMemoryFrugal* - these functions gather the training and validation data in a memory efficient way
for training and parameter selection.

The Torch directory contains the CUDA-aware solver for solver for the regressors. The scripts in the top-most directory
should provide guidance on how to generate data, solve for a regressor and then add the solution back to the final model.

All of the scripts were designed to be called from shell scripts, non-interactively. Hence the reliance upon getenv() for specifying
paths to resources. If this is not suitable, there is no reason why these cannot be defined explicitly in MATLAB. However, the solver will still require
generated data to be on disk and will output the solution to disk.

The settings included in the Solve.lua shoud work well. On a high-end machine, the solver takes only a minute or so to complete (nVidia PASCAL GPU).

Review these functions for details on model structure, and how to provide a shape model (mean shape).

## Running the code

Running "runDemo.m" should produce an example fit.

For training, Torch with CUDA is required. Please see lib/torch/SCR/SCR.lua and Solve.lua for dependancy information.

## License

This code is not to be used for commerical purposes. This code can be freely used for personal, academic and research purposes. However, we ask that any files retain our copyright notice
when redistributed. This software is provided ``as is'', with no warranty or guarantee. We accept no responsibility for any damages incurred.

The original work, described in the paper referenced above is the work of Ashton Fagg, Simon Lucey and Sridha Sridharan.

## Acknowledgements

I would firstly like to acknowledge Chen-Hsuan Lin for many useful discussions when developing this code.

I would also like to thank my adviser, Prof Simon Lucey, for his input in making this work possible.

