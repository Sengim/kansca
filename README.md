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

## Experiments
- `simulation/KAN.sh`: train KAN with simulation data for each bit.
- `simulation/Plot_and_symbolic.ipynb`: plot trained KAN by `simulation/KAN.sh`, fix input activations by linear function, and fine-tuning.
- `simulation/Train_extended_grid.ipynb`: train KAN with `grid_size=17` and fixed input activations.
- `simulation/evm_strategy.ipynb`: calculate accuracy of EVM strategy.
- `ascad/snr/*.ipynb`: plot SNR graph for each dataset.
- `ascad/interpret/KAN.sh|MLP.sh`: train KAN|MLP with ASCAD dataset.
- `ascad/interpret/KAN_snr.sh`: train KAN with snr-based PoI.
- `ascad/interpret/KAN_snr_grid3.ipynb|KAN_snr_grid17.ipynb`: train KAN with snr-based PoI and `grid_size=3|17`.
- `ascad/interpret/Sensitivity.ipynb|Sensitivity_MLP.ipynb`: plot sensitivity graph.
- `ascad/interpret/GE.ipynb`: plot guessing entropy graph.
- `ascad/unbalance_leakage/regression.ipynb`: train linear regression model that predicts trace from intermediate value.
