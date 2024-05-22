## Requirements (recommends)
- Docker (docker-compose)
- Nvidia-docker
- Make `dataset` folder and put dataset files like `kansca/dataset/ASCAD.h5`, `kansca/dataset/ascad_variable.h5`.  
    Note: This folder is mounted on `/mnt/d/Datasets` in the container. Please refer `docker/docker-compose.yaml`

## Usage
```bash
$ cd docker
$ docker compose up
# Attach container and run notebooks or scripts
```

## Note
- `exp/*.sh`: Train and evaluate in each conditions
- `plot_symbolic.ipynb`: Show symbolic functions in the KAN lib
- `ASCADv_SNR.ipynb`: Plot SNR graph in ASCADv  
    Note: since it does not contain r_in and r_out in the mask, SNR can't calculate in the ASCADf dataset.