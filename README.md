## Requirements (recommends)
- Docker (docker-compose)
- Nvidia-docker

## Usage
```bash
$ cd docker
$ docker compose up
# Attach container and run notebooks or scripts
```

## Note
- `train_DNN/KAN.py`, `eval_DNN/KAN.py`(, `run.sh`): Simply train and evaluate models
- `plot_symbolic.ipynb`: Show symbolic functions in the KAN lib
- `ASCADv_SNR.ipynb`: Plot SNR graph in ASCADv  
    Note: since it does not contain r_in and r_out in the mask, SNR can't calculate in the ASCADf dataset.