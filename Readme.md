## Installation

Add a subfolder pykan from https://github.com/KindXiaoming/pykan and install dependencies from there, should only miss h5py. I am using python 3.10 but should work with somewhat lower.

## Main

main.py contains basic script to train Kan. It is not very efficient, the smallish small model runs in ~10 minutes on GPU, other Kan implementations might be better(see Readme of pykan repo for more info). This setup reaches GE~=10 at 2000 traces