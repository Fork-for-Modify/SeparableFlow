# Environment Setup

## conda (can use flowformer's environment directly)
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install matplotlib tensorboard scipy opencv
pip install einops opencv-python pypng

## compile
compile the libs by "sh compile.sh"
- Change the environmental variable ($PATH, $LD_LIBRARY_PATH etc.), if it's not set correctly in your system environment (e.g. .bashrc). Examples are included in "compile.sh".

# Test

> download pretrained models from https://drive.google.com/file/d/1bpm0HmwcBrbyAsikTJR3qST6mAavQ60k/view

```shell
python evaluate_inr.py --model ./checkpoints/sepflow_sintel.pth --dataset sintel_inr --image_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_test_ratio_1_steps_20000_dec/ --flow_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_test_resized_flows/ --occlu_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_test_resized_occlusions/
```

# Train

```shell
python -u train_inr.py --stage=sintel_inr --weights ./checkpoints/sepflow_sintel.pth --gpu='0,1,2,3' --num_steps 100000 --batchSize 8 --testBatchSize=4 --lr 0.000125 --image_size 192 448 --wdecay 0.00001  --freeze_bn=1 --save_path='checkpoints/sintel' --gamma=0.85 --image_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_train_ratio_1_steps_20000_dec/ --flow_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_train_resized_flows/
```