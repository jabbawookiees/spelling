model=convolution
checkpoint=convolution
export CUDA_VISIBLE_DEVICES=
nohup python learning/trainer.py --model $model --epochs 10000 --data data/simple8/serialized.hdf5 --batch_size 3000 --checkpoint checkpoints/$checkpoint.ckpt --learning_rate 0.03 >> checkpoints/$checkpoint.log 2>&1 &


