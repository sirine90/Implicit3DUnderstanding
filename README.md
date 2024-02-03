# Structure-based Drug Design with Equivariant Diffusion Models

## Data Preprocessing
We follow: https://github.com/arneschneuing/DiffSBDD/tree/main in processing data from CrossDocked and Binding Moad

## Pre-trained models:
We use pre-trained models from: https://github.com/arneschneuing/DiffSBDD/tree/main
In particular, we guide the ligand-protein joint model provided by DiffBDD:
- `moad_ca_joint.ckpt`
- `moad_fullatom_joint.ckpt`
- `crossdocked_ca_joint.ckpt`


## Sampling

### Sample molecules for all pockets in the test set
`test.py` can be used to sample molecules for the entire testing set:
```bash
python test.py <checkpoint>.ckpt --conditioning <conditioning> --test_dir <bindingmoad_dir>/processed_noH/test/ --outdir <output_dir> --sanitize
```
There are different ways to determine the size of sampled molecules. 
- `conditioning`: 
- `--fix_n_nodes`: generates ligands with the same number of nodes as the reference molecule
- `--n_nodes_bias`: we add 5 nodes to CrossDocked samples and 10 to Binding Moad samples
- `--n_samples`: Number of sampled molecules
- `--timesteps`: Number of denoising timesteps for inference 
- `--resamplings` | Number of resampling steps 


### Metrics
For assessing basic molecular properties create an instance of the `MoleculeProperties` class and run its `evaluate` method:
```python
from analysis.metrics import MoleculeProperties
mol_metrics = MoleculeProperties()
all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = \
    mol_metrics.evaluate(pocket_mols)
```

For computing docking scores, run QuickVina as described below. 

### QuickVina2
First, convert all protein PDB files to PDBQT files using MGLTools
```bash
conda activate mgltools
cd analysis
python docking_py27.py <bindingmoad_dir>/processed_noH/test/ <output_dir> bindingmoad
cd ..
conda deactivate
```
QuickVina scores is computed as:
```bash
conda activate sbdd-env
python analysis/docking.py --pdbqt_dir <docking_py27_outdir> --sdf_dir <test_outdir> --out_dir <qvina_outdir> --write_csv --write_dict
```

## Citation
```
@article{schneuing2022structure,
  title={Structure-based drug design with equivariant diffusion models},
  author={Schneuing, Arne and Du, Yuanqi and Harris, Charles and Jamasb, Arian and Igashov, Ilia and Du, Weitao and Blundell, Tom and Li{\'o}, Pietro and Gomes, Carla and Welling, Max and others},
  journal={arXiv preprint arXiv:2210.13695},
  year={2022}
}
```
