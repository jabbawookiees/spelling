model=autoencoder2
checkpoint=autoencoder2
nohup python learning/trainer.py --model $model --epochs 4000 --data data/simple/serialized.hdf5 --checkpoint checkpoints/$checkpoint.ckpt --learning_rate 0.01 >> checkpoints/$checkpoint.log 2>&1 &


