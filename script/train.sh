

# CUDA_VISIBLE_DEVICES=4,5 torchrun --no_python --nproc_per_node=2 --master_port=62119 nest task run config/cod.yml -o work_dir=./output/hardmard_nc4k -o launcher=pytorch

# CUDA_VISIBLE_DEVICES=2,3 torchrun --no_python --nproc_per_node=2 --master_port=42133 
# CUDA_VISIBLE_DEVICES=0 nest task run config/sod.yml -o work_dir=./output/visual_sod -m val
# -o launcher=pytorch -m val

CUDA_VISIBLE_DEVICES=0,1 torchrun --no_python --nproc_per_node=2 --master_port=41139 nest task run config/sod.yml -o work_dir=./output/sod_04 -o launcher=pytorch 
