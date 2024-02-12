# Probabilistic Geometric Scene Reconstruction from a Single Image 
![pipeline](data/Figure.png)

## Introduction
This repo contains training, testing, evaluation and visualization code for probabilistic 3D scene reconstruction on 3D-FRONT scene renderings. 

## Install
In order to build the project, run the following:
```
sudo apt install xvfb ninja-build freeglut3-dev libglew-dev meshlab
conda env create -f Total3D.yml
conda activate Total3D
python project.py build
```

## Demo
1. First, change the username, dataroot and logroot  under ```configs/paths.py```
2. Download the [pretrained weights](https://drive.google.com/file/d/11laK4bC7wye6tTMu82v-KOBr-Mmvjewn/view?usp=share_link)
for object and layout estimation and unzip to ```out/total3d/``` and for the [probabilistic object reconstruction](https://drive.google.com/file/d/1KhNIAs1gAktPIR_Rrep76FH_mxrQtrVQ/view?usp=share_link), use the weights for conditional generation and unzip under ```out/autosdf/```

3. Change current directory to ```Implicit3DUnderstanding/``` and run the demo, which will generate 3D detection result and rendered scene mesh to ```demo/output/{scene_id}/```
    ```
    CUDA_VISIBLE_DEVICES=0 xvfb-run -a -s "-screen 0 800x600x24" python main.py out/total3d/out_config.yaml --mode demo --demo_path demo/inputs/{scene_id}
    ```

## Data preparation


#### 3D FRONT dataset
Follow  this [Link](https://drive.google.com/file/d/1QV-v3LQGvjuXdoGT_uEJcKuSe_PU9SJV/view?usp=share_link) and unzip the preprocessed training and test data to the folder data/3dfront/3dfront_train_test_data . The preprocessing follows the parametrization of [Total3DUnderstanding](https://github.com/yinyunie/Total3DUnderstanding) for 3D-FRONT scene renderings. 

#### 3D FUTURE objects for probabilistic shape reconstruction
Since 3D-FRONT is furnited with 3D-FUTURE objects, you first need to download 3D-FUTURE models from [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) to folder data/

In order to extract SDF values, we follow the preprocessing steps from [DISN](https://github.com/laughtervv/DISN)

For the generation of ground truth voxels of 3D-FUTURE objects for Resnet2VOX training, we follow [AutoSDF](https://github.com/yccyenchicheng/AutoSDF) for voxelization. 


## Training and Testing
We use [wandb](https://www.wandb.com/) for logging and visualization.
You can register a wandb account and login before training by ```wandb login```.
In case you don't need to visualize the training process, you can put ```WANDB_MODE=dryrun``` before the commands bellow.


#### Training and Evaluation 

1. Switch the keyword in 'configs/total3d.yaml' between ('layout_estimation', 'object_detection') as below to train the two tasks individually
    ```
    train:
    phase: 'layout_estimation' # or 'object_detection'
    python main.py configs/total3d.yaml --mode train
    ```
    The checkpoint can be found at ```out/total3d/[start_time]/model_best.pth```

2. Train LIEN by:
    ```
    python main.py configs/lien.yaml
    ```
    The checkpoint can be found at ```out/lien/[start_time]/model_best.pth```


3. Replace the checkpoint directories of LEN and LIEN in ```configs/total3d_lien_gcnn.yaml``` with the checkpoints trained above, then train SGCN by:
    ```
    python main.py configs/total3d_ldif_gcnn.yaml
    ```
    The checkpoint can be found at ```out/total3d/[start_time]/model_best.pth```

4. For probabilistic object reconstruction, we follow the work of [AutoSDF](https://github.com/yccyenchicheng/AutoSDF). 
As a first step, we learn the prior over 3D-FUTURE data.
We first train P-VQ-VAE on 3D-FUTURE data in order to learn the discrete latent space. Second, we use the resulting checkpoint to extract the code for every sample (in order to cache them for transformer training) by replacing ckpt with the path to the trained checkpoint and running the second command. Next, train the random-order-transformer to learn the shape prior with 3rd command. Finally, we train the image marginal module on detected 2d images:
    ```
    python train.py --model pvqvae --vq_cfg configs/pvqvae_snet.yaml
    python extract_code.py --model pvqvae --vq_cfg configs/pvqvae_snet.yaml --ckpt ${ckpt} --batch_size 1
    python train.py --model rand_tf --vq_cfg configs/pvqvae_snet.yaml --tf_cfg configs/rand_tf_snet_code.yaml --dataset_mode snet_code
    python main.py configs/model.yaml
    ```

#### Visualization of test results

1. Run the following command to evaluate the 3D reconstructions on evaluation scenes:
    ```
    python main.py out/total3d/[start_time]/out_config.yaml --mode test
    ```
    The results will be saved to ```out/total3d/[start_time]/visualization``` and the evaluation metrics will be logged to wandb as run summary.


3. Visualize the i-th 3D scene by running:
    ```
    python utils/visualize.py --result_path out/total3d/[start_time]/visualization --sequence_id [i] --save_path [] --offscreen
    ```


## Acknowledgement
This code is borrowed heavily from [Total3DUnderstanding](https://github.com/yinyunie/Total3DUnderstanding) ,[Implicit3DUnderstanding](https://github.com/chengzhag/Implicit3DUnderstanding) and [AutoSDF](https://github.com/yccyenchicheng/AutoSDF). Thanks for the efforts for making their code available!

