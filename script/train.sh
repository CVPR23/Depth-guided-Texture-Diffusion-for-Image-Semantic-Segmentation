CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/win_7.yml -o work_dir=./output/direct_add -o launcher=pytorch 
wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/win_11.yml -o work_dir=./output/win11 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/win_15.yml -o work_dir=./output/win15 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/win_22.yml -o work_dir=./output/win22 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/win_3.yml -o work_dir=./output/win3 -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/win_9.yml -o work_dir=./output/win9 -o launcher=pytorch 

