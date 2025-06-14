#!/bin/bash

#### DATASET
DATASET="patherea"
# Change dataset
DATA_DIR="$(realpath ./patherea_dataset/Patherea_LNET_224_v1.1)"
#DATA_DIR="$(realpath ./patherea_dataset/Patherea_GNET_224_v1.1)"
#DATA_DIR="$(realpath ./patherea_dataset/Patherea_Breast_P1_224_v1.1)"
#DATA_DIR="$(realpath ./patherea_dataset/Patherea_Breast_P2_224_v1.1)"

# Change TEST fold(s)
TEST_FOLDS="[1]"

#### MODEL
METHOD_SEED=""
DATA_SEED=100
LOSS_1_N_W=0.5
DET_THRESH=0.5
LEVELS="[2]"
N_ANCHORS="[2, 2]"
LOSS_SCALE=1
W_REG=2e-3
B_SIZE=16
EPOCHS=10
SAVE_EVAL_FREQ=10
BACKBONE="ViT-Adapter"
BACKBONE_VAR=B
NUM_FPN_FEATURES=768
H_P2P=True
H_MATCH_N=6
VIT_MLP=True
FREEZE_BACKBONE=False
PRETRAINED=True
HEIGHT=224
WIDTH=224
STRIDE=224
TRAIN_RATIO=1
CLASSES="[1, 2, 3, 4, 5]"
SOLO_OTHER_POS=False
NORM_MEAN="[0.485, 0.456, 0.406]"
NORM_STD="[0.229, 0.224, 0.225]"
HUNGARIAN_BOOL=True

RUN_NAME="${DATASET}_${BACKBONE}_${BACKBONE_VAR}_${NUM_FPN_FEATURES}_e${EPOCHS}_h_${H_P2P}_${H_MATCH_N}_b${B_SIZE}"

cd src
python -m p2p.run_aikinet_multiple \
        "params.desc=$DESC" \
        params.run_name=${RUN_NAME} \
        params.experiment_name=${RUN_NAME} \
        params.save_eval_freq=${SAVE_EVAL_FREQ} \
        params.log_checkpoint=False \
        dataset.data_seed=${DATA_SEED} \
        "dataset.test_folds=${TEST_FOLDS}" \
        "dataset.data_dir=${DATA_DIR}" \
        dataset.hungarian_bool=${HUNGARIAN_BOOL} \
        model.method_seed=${METHOD_SEED} \
        model.h_p2p=${H_P2P} \
        model.matching_n=${H_MATCH_N} \
        model.backbone=${BACKBONE} \
        model.backbone_variant=${BACKBONE_VAR} \
        model.pretrained=${PRETRAINED} \
        model.num_fpn_features=${NUM_FPN_FEATURES} \
        model.loss_total_1_n=${LOSS_1_N_W} \
        model.loss_total_w_reg=${W_REG} \
        model.batch_size=${B_SIZE} \
        "model.levels=${LEVELS}" \
        "model.n_anchors=${N_ANCHORS}" \
        model.loss_scale=${LOSS_SCALE} \
        "dataset.det_thresh=${DET_THRESH}" \
        dataset.epochs=$EPOCHS \
        dataset.test_stride=$STRIDE \
        model.height=$HEIGHT \
        model.width=$WIDTH \
        dataset.solo_other_pos=${SOLO_OTHER_POS} \
        dataset.train_ratio=${TRAIN_RATIO} \
        "dataset.classes=${CLASSES}" \
        "model.vit_mlp_layer=${VIT_MLP}" \
        dataset.solo_other_pos=${SOLO_OTHER_POS} \
        "model.normalize_mean=${NORM_MEAN}" \
        "model.normalize_std=${NORM_STD}" \
        "model.freeze_backbone=${FREEZE_BACKBONE}"
cd ..

echo "Finished execution!"