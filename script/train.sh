

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/15.yml -o work_dir=./output/prompt_res_embdim -o launcher=pytorch 
