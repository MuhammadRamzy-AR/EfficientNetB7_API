# EfficientNetB7 Classification Website (Backend)

This API is made to serve main EfficientNetB7 Classifaction website (front-end) to do image classification with EfficientNetB7 model.

# How to Run
## Importing requirements.txt
1.  Make sure you have python or preferably anaconda or miniconda installed.
2.  Create new virtual environment using the command below.
- Command to create a new conda environment:
```console
user@pc:~$ conda create --name flask_api --file requirements_win10.txt
```
- If you prefer to install the required packages to an existing environment, you can use this command:
```console
user@pc:~$ conda install --name flask_api --file requirements_win10.txt
```
3.  Activate the created environment.
- Command to activate the environment:
```console
user@pc:~$ conda activate flask_api
```
## Manually Install The Required Packages (Alternative)
If you prefer to install the required packages manually, you can follow the steps below.
1.  Make sure you have python or preferably anaconda or miniconda installed.
2.  Create new virtual environment to prevent breaking your main python or conda base environment.
    example for creating new environment with conda
```console
user@pc:~$ conda create --name flask_api
user@pc:~$ conda activate flask_api
```
3.  Install required packages to run the API.
```console
user@pc:~$ conda install pip
user@pc:~$ conda install flask tensorflow pillow flask-cors opencv
user@pc:~$ conda install h5py
user@pc:~$ conda install -c anaconda cudnn
```
4.  Run the API.
```console
user@pc:~$ flask run
```

## Common Troubleshooting
1.  SSL Module not Found Error
```console
SSL Module not Found
```
If you get an error that looks like the output above when trying to install a package in the environment, you can troubleshoot this problem by copy-paste the following files from conda_installation_directory\Library\bin to conda_installation_directory\DLLs.
- libcrypto-1_1-x64.dll
- libcrypto-1_1-x64.pdb
- libssl-1_1-x64.dll
- libssl-1_1-x64.pdb
\
If you leave the default option for installation directory when installing anaconda or miniconda, usually the installation directory is located at:
```console
C:\Users\<your_username>\Anaconda3\ or C:\ProgramData\Anaconda3
```
For Anaconda and for Miniconda:
```console
C:\Users\<your_username>\Miniconda3\ or C:\ProgramData\Miniconda3
```
2. ImportError on importing _imageging.
```console
ImportError: DLL load failed while importing _imaging: The specified module could not be found.
```
If you encounter this problem, you may need to upgrade pillow package. You can upgrade the package by running below command.
```console
user@pc:~$ pip install --upgrade Pillow
```

3.  Error when running the classification, (mostly when running on NVidia GPU).
```console
user@pc:~$ Could not locate zlibwapi.dll. Please make sure it is in your library path!
```
You can solve the problem by installing cudnn in your conda environment by executing the command below.
```console
user@pc:~$ user@pc:~$ conda install -c anaconda cudnn
```
