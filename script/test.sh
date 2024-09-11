CUDA_VISIBLE_DEVICES=0,1 torchrun --no_python --nproc_per_node=2 --master_port=46415 nest task run config/sod.yml -o work_dir=./output/val -o launcher=pytorch -m val
