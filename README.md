<div align="center">    
 
# TAMER:

</div>

## Project structure
```bash
├── config/         # config for TAMER hyperparameter
├── data/
│   └── crohme      # CROHME Dataset
│   └── HME100k      # HME100k Dataset which needs to be downloaded according to the instructions below.
├── eval/             # evaluation scripts
├── tamer               # model definition folder
├── lightning_logs      # training logs
│   └── version_0      # ckpt for CROHME dataset
│       ├── checkpoints
│       │   └── epoch=125-step=47375-val_ExpRate=0.6101.ckpt
│       ├── config.yaml
│       └── hparams.yaml
│   └── version_1      # ckpt for HME100k dataset
│       ├── checkpoints
│       │   └── epoch=55-step=175503-val_ExpRate=0.6924.ckpt
│       ├── config.yaml
│       └── hparams.yaml
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── train.py
```

## Install dependencies   
```bash
cd TAMER
# install project   
conda create -y -n TAMER python=3.7
conda activate TAMER
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
 ```
## Dataset Preparation
We have prepared the CROHME dataset and HME100K dataset in [download link](https://disk.pku.edu.cn/link/AAF10CCC4D539543F68847A9010C607139). After downloading, please extract it to the `data/` folder.

## Training on CROHME Dataset
Next, navigate to TAMER folder and run `train.py`. It may take **8~9** hours on **4** NVIDIA 2080Ti gpus using ddp.
```bash
# train TAMER model using 4 gpus and ddp on CROHME dataset
python -u train.py --config config/crohme.yaml
```

For single gpu user, you may change the `config.yaml` file to
```yaml
gpus: 1
```

## Training on HME100k Dataset
It may take about **48** hours on **4** NVIDIA 2080Ti gpus using ddp on HME100k dataset.
```bash
# train TAMER model using 4 gpus and ddp on hme100k dataset
python -u train.py --config config/hme100k.yaml
```

## Evaluation
Trained TAMER weight checkpoints for CROHME and HME100K Datasets have been saved in `lightning_logs/version_0` and `lightning_logs/version_1`, respectively.

```bash
# For CROHME Dataset
bash eval/eval_crohme.sh 0

# For HME100K Dataset
bash eval/eval_hme100k.sh 1
```


## Reference
- [CoMER](https://github.com/Green-Wood/CoMER) | [arXiv](https://arxiv.org/abs/2207.04410)
- [BTTR](https://github.com/Green-Wood/BTTR) | [arXiv](https://arxiv.org/abs/2105.02412)
- [TreeDecoder](https://github.com/JianshuZhang/TreeDecoder)
- [CAN](https://github.com/LBH1024/CAN) | [arXiv](https://arxiv.org/abs/2207.11463)

