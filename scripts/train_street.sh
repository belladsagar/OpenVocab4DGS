#!/bin/bash
# chmod +x scripts/train_scannet.sh
# ./scripts/train_scannet.sh

# ============== [Notice] ==============
# 1. The 10 scene hyperparameters in the ScanNet dataset are consistent.
# 2. Train a scene for about 20 minutes on a 24G 4090 GPU.
# 3. Please check the dataset path specified by -s.

# ============== [Hyperparameter explanation] ==============
# Total training steps: 90k
# 3dgs pre-train: 0~30k
# stage1: 30~50k
# stage2 (coarse-level): 50~70k
# stage2 (fine-level): 70k~90k
# k1=64, k2=5
# frozen_init_pts: The point clouds provided by the ScanNet dataset are frozen, without using the densification scheme of 3DGS.
# -r 2 : We use half-resolution data for training.

# ============== [10 scenes] ==============
scan_list=("031" )

gpu_num=0     # change!
for scan in "${scan_list[@]}"; do
    echo "Training for ${scan} ....."
    CUDA_VISIBLE_DEVICES=$gpu_num python train.py --port 601$gpu_num \
        -s /data/riu/waymo/waymo/processed/training/${scan} \
        --start_checkpoint output/waymo_full_exp_1/waymo_train_031/trained_model/iteration_50000.pth \
        -r 2 \
        --frozen_init_pts \
        --iterations 40_000 \
        --start_ins_feat_iter 0 \
        --start_root_cb_iter 20_000 \
        --start_leaf_cb_iter 20_000 \
        --sam_level 3 \
        --root_node_num 64 \
        --leaf_node_num 5 \
        --pos_weight 1.0 \
        --test_iterations 30000 \
        --eval
done