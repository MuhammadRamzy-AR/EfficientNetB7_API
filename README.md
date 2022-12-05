# ResNet 152 Classification Website (Backend)

This API is made to serve main ResNet 152 Classifaction website (front-end) to do image classification with ResNet-152 V1 and ResNet-152 V2 model.

## How to Run
1.  Make sure you have python or preferably anaconda or miniconda installed.
2.  Create new virtual environment to prevent breaking your main python or conda base environment.
    example for creating new environment with conda
```console
user@pc:~$ conda create --name flask_api
user@pc:~$ conda activate flask_api
```
3.  Install required packages to run the API.
```console
user@pc:~$ pip install flask tensorflow pillow flask-cors opencv-python
user@pc:~$ conda install h5py
user@pc:~$ conda install -c anaconda cudnn
```
4.  Run the API.
```console
user@pc:~$ flask run
```

## Common Troubleshooting
1.  ImportError on importing _imageging.
```console
ImportError: DLL load failed while importing _imaging: The specified module could not be found.
```
If you encounter this problem, you may need to upgrade pillow package. You can upgrade the package by running below command.
```console
user@pc:~$ pip install --upgrade Pillow
```

2.  Error when running the classification, (mostly when running on NVidia GPU).
```console
user@pc:~$ Could not locate zlibwapi.dll. Please make sure it is in your library path!
```
You can solve the problem by installing cudnn in your conda environment by executing the command below.
```console
user@pc:~$ user@pc:~$ conda install -c anaconda cudnn
```
