

CUDA_VISIBLE_DEVICES=2,3 torchrun --no_python --nproc_per_node=2 --master_port=46415 nest task run config/15.yml -o work_dir=./output/prompt_res_embed_dim -o launcher=pytorch 

