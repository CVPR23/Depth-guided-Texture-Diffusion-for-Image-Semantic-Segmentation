

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/1.yml -o work_dir=./output/iter4 -o launcher=pytorch 
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/2.yml -o work_dir=./output/iter3 -o launcher=pytorch 
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/3.yml -o work_dir=./output/iter2 -o launcher=pytorch 
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/4.yml -o work_dir=./output/iter6 -o launcher=pytorch 
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/5.yml -o work_dir=./output/iter7 -o launcher=pytorch 
wait 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/6.yml -o work_dir=./output/iter8 -o launcher=pytorch 
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/7.yml -o work_dir=./output/iter9 -o launcher=pytorch 
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/8.yml -o work_dir=./output/iter10 -o launcher=pytorch 
wait 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/9.yml -o work_dir=./output/iter11 -o launcher=pytorch 
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/10.yml -o work_dir=./output/iter12 -o launcher=pytorch 
wait 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/11.yml -o work_dir=./output/no_propagation -o launcher=pytorch 

