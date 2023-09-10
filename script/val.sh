# CUDA_VISIBLE_DEVICES=4 nest task run config/val_win_3.yml -o work_dir=output/val_win_3 -m val &
# CUDA_VISIBLE_DEVICES=5 nest task run config/val_win_5.yml -o work_dir=output/val_win_5 -m val &
# CUDA_VISIBLE_DEVICES=6 nest task run config/val_win_7.yml -o work_dir=output/val_win_7 -m val &
# CUDA_VISIBLE_DEVICES=7 nest task run config/val_win_9.yml -o work_dir=output/val_win_9 -m val &
# wait
# CUDA_VISIBLE_DEVICES=4 nest task run config/val_win_11.yml -o work_dir=output/val_win_11 -m val &
# CUDA_VISIBLE_DEVICES=5 nest task run config/val_win_13.yml -o work_dir=output/val_win_13 -m val &
# CUDA_VISIBLE_DEVICES=6 nest task run config/val_win_15.yml -o work_dir=output/val_win_15 -m val &
# CUDA_VISIBLE_DEVICES=7 nest task run config/val_win_17.yml -o work_dir=output/val_win_17 -m val &
CUDA_VISIBLE_DEVICES=4 nest task run config/val_max_09.yml -o work_dir=output/val_max_09 -m val &
CUDA_VISIBLE_DEVICES=5 nest task run config/val_max_08.yml -o work_dir=output/val_max_08 -m val &
CUDA_VISIBLE_DEVICES=6 nest task run config/val_max_07.yml -o work_dir=output/val_max_07 -m val &
CUDA_VISIBLE_DEVICES=7 nest task run config/val_max_06.yml -o work_dir=output/val_max_06 -m val &
wait
CUDA_VISIBLE_DEVICES=4 nest task run config/val_max_05.yml -o work_dir=output/val_max_05 -m val &
CUDA_VISIBLE_DEVICES=5 nest task run config/val_max_04.yml -o work_dir=output/val_max_04 -m val &
CUDA_VISIBLE_DEVICES=6 nest task run config/val_max_03.yml -o work_dir=output/val_max_03 -m val &
CUDA_VISIBLE_DEVICES=7 nest task run config/val_max_02.yml -o work_dir=output/val_max_02 -m val &
