

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/15.yml -o work_dir=./output/iter3_3layer_trial2 -o launcher=pytorch 

