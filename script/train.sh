
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/no_prompt_dpt_boosted.yml -o work_dir=./output/new_prompt_dpt_nyu_boosted -o launcher=pytorch 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --no_python --nproc_per_node=4 --master_port=46415 nest task run config/no_prompt.yml -o work_dir=./output/x_iy_ix -o launcher=pytorch 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --no_python --nproc_per_node=8 --master_port=46415 nest task run config/no_prompt_60epo.yml -o work_dir=./output/60_scratch_full -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --no_python --nproc_per_node=8 --master_port=46415 nest task run config/no_prompt_80epo.yml -o work_dir=./output/80_scratch_full -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --no_python --nproc_per_node=8 --master_port=46415 nest task run config/no_prompt_hit.yml -o work_dir=./output/20_hitnet_full -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --no_python --nproc_per_node=8 --master_port=46415 nest task run config/no_prompt_hit_60epo.yml -o work_dir=./output/60_hitnet_full -o launcher=pytorch 
# wait
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --no_python --nproc_per_node=8 --master_port=46415 nest task run config/no_prompt_hit_80epo.yml -o work_dir=./output/80_hitnet_full -o launcher=pytorch 



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --no_python --nproc_per_node=8 --master_port=46415 nest task run config/kitti.yml -o work_dir=./output/test_kitti -o launcher=pytorch 
# python BoostYourOwnDepth/boost_depth.py --data_dir /root/autodl-tmp/dataset/cod10k_test/Depth_boost_dpt_nyu/ --output_dir /root/autodl-tmp/dataset/cod10k_test/Depth_boost_dpt_nyu/boosted/ --checkpoints_dir BoostYourOwnDepth/pix2pix/checkpoints --colorize_results

