# Ligand-Based Drug Design using UniGuide Framework

### Additional Requirements

oddt - 0.7

pytorch3d - 0.7.1

Other packages include tqdm, yaml, lmdb.

### Data Preprocessing
Download and extract the MOSES dataset as described by the authors of ShapeMol: https://arxiv.org/abs/2308.11890 

### Training the Unconditional Equivariant Diffusion Model for 3D molecules

Please use the command below to train the diffusion model:
```bash
python -m scripts.train_diffusion ./config/training/unconditional_shapemol.yml --logdir <path to save trained models>
```

### Test

We provided our trained model in the directory "trained_models".

Please use the command below to generate ligands given a test reference ligand:
```bash
python -m scripts.sample_diffusion ./config/sampling/ --result_path ./result/uniguide/ --data_id 0
```
where data_id is selected from [0,999], and it refers to the index of reference ligand.

### Analyze Results 

We also provided a jupyter notebook in /notebooks/Analyze_results.ipynb to visualize all the generated molecules and analyze their properties.

## Citation
```
@article{chen2023shape,
  title={Shape-conditioned 3D Molecule Generation via Equivariant Diffusion Models},
  author={Chen, Ziqi and Peng, Bo and Parthasarathy, Srinivasan and Ning, Xia},
  journal={arXiv preprint arXiv:2308.11890},
  year={2023}
}
```
