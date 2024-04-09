# ResNet Training on CIFAR-10 with ColossalAI
This project shows how to train ResNet models on the CIFAR-10 dataset from scratch using ColossalAI. Includes examples of FP32 training, mixed precision training, and low-level zero replication training.

### Github link
https://github.com/871924643/ResNet-18-with-ColossalAI

### Install requirements

```bash
pip install -r requirements.txt
```

### Train
The folders will be created automatically.
```bash
# train with torch DDP with fp32
colossalai run --nproc_per_node 1 train.py -c ./ckpt-fp32

# train with torch DDP with mixed precision training
colossalai run --nproc_per_node 1 train.py -c ./ckpt-fp16 -p torch_ddp_fp16

# train with low level zero
colossalai run --nproc_per_node 1 train.py -c ./ckpt-low_level_zero -p low_level_zero
```

### Eval

```bash
# evaluate fp32 training
python eval.py -c ./ckpt-fp32 -e 10

# evaluate fp16 mixed precision training
python eval.py -c ./ckpt-fp16 -e 30

# evaluate low level zero training
python eval.py -c ./ckpt-low_level_zero -e 30
```
### How to run the code
Please refer to log.ipynb to train and evaluate the model.
### Model
ResNet-18
### Dataset

The CIFAR-10 dataset was used for this experiment.
### Parallel Setups

This experiment provides three example configurations for parallel training, including:

Torch DDP (FP32)
Torch DDP (FP16 mixed precision)
Low-level zero replication

### Accuracy performance:

| Training configuration     | Epoch 10 | Epoch 30 |
| --------- | ------------------------ | ---------------------|
| FP32| 74.66%                   | -                |
| FP16 Mixing Precision| 68.15%	 | 75.84%           |
| Low-level Zero| 67.28%	       | 76.65%           |



