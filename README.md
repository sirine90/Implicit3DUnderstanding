# UniGuide: Unified Guidance for Geometry-Conditioned Molecular Generation

# Setup
First, please create a new environment by running 
```
conda env create -f environment.yml
```

# QuickVina 2
For measuring Vina score, intall QuickVina 2:
```bash
wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1
chmod +x qvina2.1 
```

In order to prepare the receptor for docking, we recommend to create a new environment:
```bash
conda create -n mgltools -c bioconda mgltools
```


