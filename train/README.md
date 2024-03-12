# Method of UniGarmentManip
![Method](../image/method.png)
Our Proposed Learning Framework for Dense Visual Correspondence. We extract the cross-deform correspondence and cross-object correspondence point pairs respectively using self-play and skeletons, and train the per-point correspondence scores in the contrastive manner, with the Coarse-to-fine module refines the quality. Learned correspondence demonstrates point-level similarity across different garments in different deformations. The learned point-level correspondence can facilitates multiple diverse downstream tasks using one or few-shot demonstrations.

## Installation
1. You need to install conda environment in the main page
2. Install CUDA 11.0 through the following command
```bash
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.runsudo sh cuda_11.0.2_450.51.05_linux.run
```
3. You need to install [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch) using the following command
```bash
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .
```
4. Trouble shooting<br>
   (1) If you encounter compile error when compiling PointNet++. We recommend you add following code into .bashrc
    ```bash
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
    ```
   (2) You can change cuda version use the soft link, for example,
    ```bash
   sudo rm -rf /usr/local/cuda
    sudo ln -s /usr/local/cuda-11.0 /usr/local/cuda
    ```
## Usage
1. Prepare data<br>
    Note that the data collected in the simulator is stored as pkl format. However, due to the loading speed, we convert the pkl file into h5 file. You can use the following command to convert the pkl file into h5 file
    ```bash
   python train/dataloader/preprocess/pickle2h5.py
2. Start Training process<br>
   You can start training process through the following command
   ```bash
   python train/train/avertrain.py # coarse training process
   python train/train/c2f.py # coarse to fine refinement process
   python train/train/distrain.py # distance train
   ```
3. Training Process
   (1) You can change the hyperparameter in the base folder
   (2) You can change the structure of model and dataloader in the corresponding folder
