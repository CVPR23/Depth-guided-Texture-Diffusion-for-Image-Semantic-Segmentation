# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/16.yml -o work_dir=./output/360prompt_baseline -o launcher=pytorch
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/17.yml -o work_dir=./output/256prompt_baseline -o launcher=pytorch
# wait

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/1.yml -o work_dir=./output/sod_baseline -o launcher=pytorch
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/1.yml -o work_dir=./output/sod_step1 -o launcher=pytorch 
# wait

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46414 nest task run config/2.yml -o work_dir=./output/cod_prompt_embeddim_cross_22 -o launcher=pytorch 

# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/3.yml -o work_dir=./output/sod_step3 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/4.yml -o work_dir=./output/sod_step4 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/5.yml -o work_dir=./output/sod_step5 -o launcher=pytorch -m val 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/6.yml -o work_dir=./output/sod_step6 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/7.yml -o work_dir=./output/sod_step7 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/8.yml -o work_dir=./output/sod_step8 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/9.yml -o work_dir=./output/sod_step9 -o launcher=pytorch 



# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/4.yml -o work_dir=./output/iter4_256prompt -o launcher=pytorch
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/5.yml -o work_dir=./output/iter5_256prompt -o launcher=pytorch
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/6.yml -o work_dir=./output/iter6_256prompt -o launcher=pytorch
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/7.yml -o work_dir=./output/iter7_256prompt -o launcher=pytorch
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/8.yml -o work_dir=./output/iter8_256prompt -o launcher=pytorch
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/9.yml -o work_dir=./output/iter9_256prompt -o launcher=pytorch
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/14.yml -o work_dir=./output/iter1_lr001 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/16.yml -o work_dir=./output/iter1 -o launcher=pytorch 
# wait

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/15.yml -o work_dir=./output/iter3_3layer_trial2 -o launcher=pytorch 
# wait


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/13.yml -o work_dir=./output/iter3_latent_dim_48 -o launcher=pytorch 

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/17.yml -o work_dir=./output/iter2 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/18.yml -o work_dir=./output/iter4 -o launcher=pytorch 
# wait

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/7.yml -o work_dir=./output/iter6 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/8.yml -o work_dir=./output/iter7 -o launcher=pytorch 

