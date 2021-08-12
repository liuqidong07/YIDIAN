#! /bin/bash
model_name='dnn'
batch_size=8192
learning_rate=0.001
epoch=1
device_tab=2
embedding_dim=16
lambda=0.1
#CUDA_LAUNCH_BLOCKING=1 python main.py -m ${model_name} -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim}
#CUDA_LAUNCH_BLOCKING=1 python grid_search.py -m esmm -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log
CUDA_LAUNCH_BLOCKING=1 python main_MT.py -m esmm -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -labda ${lambda} -online
CUDA_LAUNCH_BLOCKING=1 python main_MT.py -m esmm -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -labda ${lambda} -online
CUDA_LAUNCH_BLOCKING=1 python main_MT.py -m esmm -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -labda ${lambda} -online
CUDA_LAUNCH_BLOCKING=1 python main_MT.py -m esmm -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -labda ${lambda} -online
CUDA_LAUNCH_BLOCKING=1 python main_MT.py -m esmm -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -labda ${lambda} -online
