# Latent Disentanglement in Mesh Variational Autoencoders Improves the Diagnosis of Craniofacial Syndromes and Aids Surgical Planning

This repo extends the **3D Shape Variational Autoencoder Latent Disentanglement 
via Mini-Batch Feature Swapping for Bodies and Faces** code. In particular, 
it provides additional functionalities that can be used for the diagnosis 
and surgical planning of Crouzon, Apert and Muenke craniofacial syndromes. 
If you use this code, please cite also the aforementioned paper.
  
  
## System requirements

This software was implemented on Ubuntu 20.04 LTS, with Python 3.8, and CUDA 10.1.
The software was also tested on Ubuntu 20.04 LTS, with Python 3.8, and CUDA 11.3.
Different OSs can be used as long as they have a pytorch-compatible GPU and 
python is available. Different versions of CUDA and Python can be used. 
Follow the installation instructions to make sure that the correct libraries are
installed. All the necessary dependencies will be automatically installed 
following the installation instructions.
The version of each library is specified in the `install_env.sh` script. 
Even though newer versions are likely to work, we guarantee the correct
functioning of the code only with the specified libraries. 


## Installation

After cloning the repo, open a terminal and go to the project directory. 

Change the permissions of install_env.sh by running `chmod +x ./install_env.sh` 
and run it with:
```shell script
./install_env.sh
```
This will create a virtual environment with all the necessary libraries. 
The automatic installation should take approximately 7 minutes. Note that
pytorch is very heavy and if a pre-existing version is not pre-cached, downloading 
pytorch may take up to 1 hour. The download speed is just a rough estimates 
because it is highly affected by the internet connection.

Note that the code was developed with Python 3.8, CUDA 10.1, and Pytorch 1.7.1. 
The code was also tested with Python 3.8, CUDA 11.3, and Pytorch 1.12.1. 
We provide an automatic installation script for the latest set of libraries. 
The code should work also with other versions of  Python, CUDA, and Pytorch. 
If you wish to try running the code with more recent versions of these libraries, 
change the CUDA, TORCH, TORCHVISION, and PYTHON_V variables in `install_env.sh`

If you want to install CUDA 11.3 and you already have another CUDA version, 
we recommend following the instruction provided at https://jin-zhe.github.io/guides/getting-cudnn/.
Obviously, make sure that the commands you type in the terminal have the correct 
CUDA version and not the one suggested in the tutorial. Before installing a new 
CUDA version, check the drivers compatibility at https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html.

Then activate the virtual environment :
```shell script
source ./id-generator-env/bin/activate
```


## Demo notebook
`demo.ipynb` contains a demo that even people without access to the original 
dataset can use. 
The demo will allow you to test the data augmentation, 
classify patients, project them on the global and local latent distributions, as
well as to simulate ideal surgical procedures. The expected outputs of the 
different cells are a reproduction of the images of the paper on 
different patients.

Running all the cells of the demo code should take approximately 5 minutes on a 
laptop. Most of the time is used by the eigenvalue decomposition for the data 
augmentation. The respective cell is not necessary for the correct execution of 
the others and it can be skipped if you are not interested in demoing the 
spectral interpolation.

To run the demo download the `demo_files` folder and copy it in the project 
directory. The demo files within the `demo_files` folder are:
 - the precomputed down- and up-sampling transformations,
 - the precomputed spirals,
 - the mesh template segmented according to the different anatomical sub-units,
 - the network weights,
 - the pre-trained global LDA and QDA,
 - the pre-trained local LDAs,
 - the precomputed plots for local and global distributions,
 - the data normalisation values,
 - a subset of synthetic meshes obtained with our data-augmentation. Note that these meshes are not of real subjects.  
 
 
 ## Changes required to train the model on other datasets
 
 We made available the configuration file used in the experiments 
 (`craniofacial.yaml`). If you want to train SD-VAE on your own dataset make
 sure that the paths in the config file are correct and that the to_mm_constant 
 is correct. 
 
 Currently meshes in the dataset are labelled according to the first letter of 
 their filename. If you want to use our dataloaders make sure your meshes are 
 named according to the classes of your problem (NB. at the moment meshes with 
 the name starting with "n" and "b" are grouped together).
 The data_summary file is an xlsx file currently used to determine which meshes 
 should be loaded and other information such ase patient's age and gender. 
 The most important functions that you may have to modify to train on a new 
 datased are: `_process_set` in *data_generation_and_loading.py* as well as 
 `get_dataset_summary`, `find_data_used_from_summary`, and 
 `get_age_and_gender_from_summary` in *utils.py*.
  
 
 ## Train and Test
 
 To start the training from the project repo simply run:
 ```shell script
python train.py --config=configurations/craniofacial.yaml --id=<NAME_OF_YOUR_EXPERIMENT>
```

Basic tests will automatically run at the end of the training. These tests will 
run on your validation set. If you wish to run experiments on the test set or if 
you want to perform only a subset of the experiments presented in the paper 
you can uncomment any function call at the end of `test.py`. If your model has 
alredy been trained or you are using our pretrained model, you can run tests 
without training:
```shell script
python test.py --id=<NAME_OF_YOUR_EXPERIMENT>
```
Note that NAME_OF_YOUR_EXPERIMENT is also the name of the folder containing the
pretrained model.

