#!/bin/bash
# ==============================================================================
# FiLM++ Soft Experiments
# ==============================================================================

set -e  # Exit on error

# Base paths
DATA_ROOT="/home/AIoT04/Datasets/icbhi_dataset"
AST_PRETRAINED="/home/AIoT04/Dev/pretrained_models/audioset_10_10_0.4593.pth"
OUTPUT_DIR="outputs/filmpp_experiments"
GPU_ID=1

# Common settings
COMMON_ARGS="
    --data_root ${DATA_ROOT}
    --ast_pretrained ${AST_PRETRAINED}
    --output_dir ${OUTPUT_DIR}
    --batch_size 16
    --num_workers 0
    --weighted_sampler
    --epochs 40
    --grad_clip 1.0
    --warmup_epochs 3
    --seed 42
    --device cuda
    --save_best
"

# ==============================================================================
# Experiment 1: Baseline FiLM++ with different conditioned layers
# ==============================================================================

echo "====================================================================="
echo "Experiment 1: Varying Conditioned Layers"
echo "====================================================================="

# Last 2 layers only
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 10 11 \
    --dev_emb_dim 4 \
    --site_emb_dim 7 \
    --metadata_hidden_dim 32 \
    --film_hidden_dim 32 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --exp_name layers_10_11

# Last 4 layers
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 8 9 10 11 \
    --dev_emb_dim 4 \
    --site_emb_dim 7 \
    --metadata_hidden_dim 32 \
    --film_hidden_dim 32 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --exp_name layers_8_9_10_11

# All layers
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 0 1 2 3 4 5 6 7 8 9 10 11 \
    --dev_emb_dim 4 \
    --site_emb_dim 7 \
    --metadata_hidden_dim 32 \
    --film_hidden_dim 32 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --exp_name layers_all

# ==============================================================================
# Experiment 2: Mask Sparsity Lambda
# ==============================================================================

echo "====================================================================="
echo "Experiment 2: Varying Mask Sparsity Lambda"
echo "====================================================================="

for LAMBDA in 0.0 0.001 0.01 0.05 0.1; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
        --conditioned_layers 10 11 \
        --dev_emb_dim 4 \
        --site_emb_dim 7 \
        --metadata_hidden_dim 32 \
        --film_hidden_dim 32 \
        --mask_init_scale 2.0 \
        --mask_sparsity_lambda ${LAMBDA} \
        --lr 3e-4 \
        --weight_decay 0.05 \
        --dropout 0.5 \
        --exp_name sparsity_lambda_${LAMBDA}
done

# ==============================================================================
# Experiment 3: Mask Initialization Scale
# ==============================================================================

echo "====================================================================="
echo "Experiment 3: Varying Mask Init Scale"
echo "====================================================================="

for SCALE in 0.5 1.0 2.0 3.0 5.0; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
        --conditioned_layers 10 11 \
        --dev_emb_dim 4 \
        --site_emb_dim 7 \
        --metadata_hidden_dim 32 \
        --film_hidden_dim 32 \
        --mask_init_scale ${SCALE} \
        --mask_sparsity_lambda 0.01 \
        --lr 3e-4 \
        --weight_decay 0.05 \
        --dropout 0.5 \
        --exp_name mask_init_scale_${SCALE}
done

# ==============================================================================
# Experiment 4: Per-Layer vs Shared Masks
# ==============================================================================

echo "====================================================================="
echo "Experiment 4: Per-Layer vs Shared Masks"
echo "====================================================================="

# Shared masks (default)
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 8 9 10 11 \
    --dev_emb_dim 4 \
    --site_emb_dim 7 \
    --metadata_hidden_dim 32 \
    --film_hidden_dim 32 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --exp_name masks_shared

# Per-layer masks
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 8 9 10 11 \
    --dev_emb_dim 4 \
    --site_emb_dim 7 \
    --metadata_hidden_dim 32 \
    --film_hidden_dim 32 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --per_layer_masks \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --exp_name masks_per_layer

# ==============================================================================
# Experiment 5: Embedding Dimensions
# ==============================================================================

echo "====================================================================="
echo "Experiment 5: Varying Embedding Dimensions"
echo "====================================================================="

# Small embeddings
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 10 11 \
    --dev_emb_dim 2 \
    --site_emb_dim 4 \
    --metadata_hidden_dim 16 \
    --film_hidden_dim 16 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --exp_name emb_small

# Medium embeddings (default)
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 10 11 \
    --dev_emb_dim 4 \
    --site_emb_dim 7 \
    --metadata_hidden_dim 32 \
    --film_hidden_dim 32 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --exp_name emb_medium

# Large embeddings
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 10 11 \
    --dev_emb_dim 8 \
    --site_emb_dim 16 \
    --metadata_hidden_dim 64 \
    --film_hidden_dim 64 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --exp_name emb_large

# ==============================================================================
# Experiment 6: Learning Rate
# ==============================================================================

echo "====================================================================="
echo "Experiment 6: Varying Learning Rate"
echo "====================================================================="

for LR in 1e-4 3e-4 5e-4 1e-3; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
        --conditioned_layers 10 11 \
        --dev_emb_dim 4 \
        --site_emb_dim 7 \
        --metadata_hidden_dim 32 \
        --film_hidden_dim 32 \
        --mask_init_scale 2.0 \
        --mask_sparsity_lambda 0.01 \
        --lr ${LR} \
        --weight_decay 0.05 \
        --dropout 0.5 \
        --exp_name lr_${LR}
done

# ==============================================================================
# Experiment 7: Dropout
# ==============================================================================

echo "====================================================================="
echo "Experiment 7: Varying Dropout"
echo "====================================================================="

for DROP in 0.3 0.5 0.7; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
        --conditioned_layers 10 11 \
        --dev_emb_dim 4 \
        --site_emb_dim 7 \
        --metadata_hidden_dim 32 \
        --film_hidden_dim 32 \
        --mask_init_scale 2.0 \
        --mask_sparsity_lambda 0.01 \
        --lr 3e-4 \
        --weight_decay 0.05 \
        --dropout ${DROP} \
        --exp_name dropout_${DROP}
done

# ==============================================================================
# Experiment 8: Frozen Backbone
# ==============================================================================

echo "====================================================================="
echo "Experiment 8: Frozen vs Unfrozen Backbone"
echo "====================================================================="

# Unfrozen (default)
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 10 11 \
    --dev_emb_dim 4 \
    --site_emb_dim 7 \
    --metadata_hidden_dim 32 \
    --film_hidden_dim 32 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --exp_name backbone_unfrozen

# Fully frozen backbone
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 10 11 \
    --dev_emb_dim 4 \
    --site_emb_dim 7 \
    --metadata_hidden_dim 32 \
    --film_hidden_dim 32 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --freeze_backbone \
    --exp_name backbone_frozen

# Partially frozen (first 6 blocks)
CUDA_VISIBLE_DEVICES=${GPU_ID} python filmpp_train.py ${COMMON_ARGS} \
    --conditioned_layers 10 11 \
    --dev_emb_dim 4 \
    --site_emb_dim 7 \
    --metadata_hidden_dim 32 \
    --film_hidden_dim 32 \
    --mask_init_scale 2.0 \
    --mask_sparsity_lambda 0.01 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --dropout 0.5 \
    --freeze_backbone \
    --freeze_until_block 5 \
    --exp_name backbone_partial_frozen

# ==============================================================================
# Summary
# ==============================================================================

echo "====================================================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "====================================================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To analyze results, run:"
echo "  python analyze_experiments.py --output_dir ${OUTPUT_DIR}"