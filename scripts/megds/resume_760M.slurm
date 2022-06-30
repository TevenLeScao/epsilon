#!/bin/bash
#SBATCH --job-name=epsilon-760M
#SBATCH --qos=qos_gpu-dev           #qos_gpu-t3
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH -C v100-32g
#SBATCH --time 00:08:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/six/uhk85as/code/epsilon/logs/tests/%x-%j.out           # -output=logs/exp1/760M/%x-%j.out
#SBATCH --account=ajs@v100
#SBATCH --array=0-0                                 # 0-2735

set -e

#source $six_ALL_CCFRWORK/start-py38-pt110
#source $six_ALL_CCFRWORK/start-py38-pt111
source $six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience/train/tr11-176B-ml/start-tr11-176B-ml

echo "START TIME: $(date)"

variant=main

DATA_OUTPUT_PATH=$SCRATCH/epsilon/checkpoints/tr11d-760M-ml
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/tr11d-760M-ml-logs
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant
LOGS_PATH=$REPO_PATH/logs/$variant
mkdir -p $LOGS_PATH

MEGATRON_DEEPSPEED_REPO=$WORK/code/Megatron-DeepSpeed
EPSILON_REPO=$WORK/code/epsilon
cd $MEGATRON_DEEPSPEED_REPO

EXISTING_DATA_PATH=$WORK/code/epsilon/subdataset_1.jsonl
EXISTING_VALIDSET_DATA_PATH=$WORK/code/epsilon/subdataset_2.jsonl

SYNTHETIC_DATA_SCRIPT=$EPSILON_REPO/synthetic_datasets/size_comparison.py

SYNTHETIC_TRAIN_DATA_PATH=$DATA_OUTPUT_PATH/train_exp${SLURM_ARRAY_TASK_ID}.jsonl
SYNTHETIC_SENTENCE_DATA_PATH=$DATA_OUTPUT_PATH/valid_sentence_exp${SLURM_ARRAY_TASK_ID}.jsonl
SYNTHETIC_CORRUPTED_DATA_PATH=$DATA_OUTPUT_PATH/valid_corrupted_exp${SLURM_ARRAY_TASK_ID}.jsonl
SYNTHETIC_SUPPORT_DATA_PATH=$DATA_OUTPUT_PATH/valid_support_exp${SLURM_ARRAY_TASK_ID}.jsonl
SYNTHETIC_COUNTER_DATA_PATH=$DATA_OUTPUT_PATH/valid_counter_exp${SLURM_ARRAY_TASK_ID}.jsonl

TRAIN_DATA_PATH=$DATA_OUTPUT_PATH/train_exp${SLURM_ARRAY_TASK_ID}
SENTENCE_DATA_PATH=$DATA_OUTPUT_PATH/valid_sentence_exp${SLURM_ARRAY_TASK_ID}
CORRUPTED_DATA_PATH=$DATA_OUTPUT_PATH/valid_corrupted_exp${SLURM_ARRAY_TASK_ID}
SUPPORT_DATA_PATH=$DATA_OUTPUT_PATH/valid_support_exp${SLURM_ARRAY_TASK_ID}
COUNTER_DATA_PATH=$DATA_OUTPUT_PATH/valid_counter_exp${SLURM_ARRAY_TASK_ID}
VALIDSET_DATA_PATH=$DATA_OUTPUT_PATH/common_validset

TOKENIZER_NAME_OR_PATH=bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

# defining the right environment variables
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# data preparation  - the core of this experiment

python $SYNTHETIC_DATA_SCRIPT --single_index $SLURM_ARRAY_TASK_ID  --existing_data_path $EXISTING_DATA_PATH \
--train_filepath $SYNTHETIC_TRAIN_DATA_PATH \
--valid_sentence_filepath $SYNTHETIC_SENTENCE_DATA_PATH \
--valid_corrupted_filepath $SYNTHETIC_CORRUPTED_DATA_PATH \
--valid_support_filepath $SYNTHETIC_SUPPORT_DATA_PATH \
--valid_counter_filepath $SYNTHETIC_COUNTER_DATA_PATH \
--valid_min_size 50000 \
--override_seed_with_random                 # For tests < 2735 with random seeds

./$EPSILON_REPO/scripts/tokenize.sh $SYNTHETIC_TRAIN_DATA_PATH $TRAIN_DATA_PATH $TOKENIZER_NAME_OR_PATH
./$EPSILON_REPO/scripts/tokenize.sh $SYNTHETIC_SENTENCE_DATA_PATH $SENTENCE_DATA_PATH $TOKENIZER_NAME_OR_PATH
./$EPSILON_REPO/scripts/tokenize.sh $SYNTHETIC_CORRUPTED_DATA_PATH $CORRUPTED_DATA_PATH $TOKENIZER_NAME_OR_PATH
./$EPSILON_REPO/scripts/tokenize.sh $SYNTHETIC_SUPPORT_DATA_PATH $TOKENIZER_NAME_OR_PATH $TOKENIZER_NAME_OR_PATH
./$EPSILON_REPO/scripts/tokenize.sh $SYNTHETIC_COUNTER_DATA_PATH $COUNTER_DATA_PATH $TOKENIZER_NAME_OR_PATH
# always the same, so checking before re-creating
if [ ! -f $VALIDSET_DATA_PATH ]; then
    ./$EPSILON_REPO/scripts/tokenize.sh $EXISTING_VALIDSET_DATA_PATH $VALIDSET_DATA_PATH $TOKENIZER_NAME_OR_PATH
fi


# testing for potential faulty nodes
# srun --jobid $SLURM_JOBID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES

PP_SIZE=2
TP_SIZE=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=256
NLAYERS=24
NHIDDEN=1536
NHEADS=16
SEQ_LEN=2048
SAVE_INTERVAL=250
TRAIN_SAMPLES=100  # 450B tokens
LR_DECAY_SAMPLES=200_000_000  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=183_105  # 375M tokens
OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 2.5e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "
# for 20h 1190, for 100h 5990
#    --exit-duration-in-mins 1190 \
EXIT_OPTS=" \
    --exit-duration-in-mins 5990 \
    "
GPT_ARGS=" \
    --pp-partition-method 'type:transformer|embedding' \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 192 16 9_765_625 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --init-method-std 0.0048 \
    --embed-layernorm \
    --fp16 \
    --seed 42 \
    --position-embedding-type alibi \
    --abort-on-unmet-fused-kernel-constraints \
    --pad-vocab-size-to 250880 \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "

# TODO: decide on efficient eval-interval + eval-iters

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1 \
    --eval-iters 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=1 # this 0 by error (stage=0 means stage=1 on A100s with bf16, but it should be 1 on V100s. Will fix as soon as checkpoint reshaping is possible)

config_json="./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --train-weighted-split-paths 'train: 1 0:1 ${TRAIN_DATA_PATH}_text_document' \
    --valid-weighted-split-paths \
    'sentence: 1 0:1 ${SENTENCE_DATA_PATH}_text_document' \
    'corrupted: 1 0:1 ${CORRUPTED_DATA_PATH}_text_document' \
    'support: 1 0:1 ${SUPPORT_DATA_PATH}_text_document' \
    'counter: 1 0:1 ${COUNTER_DATA_PATH}_text_document' \
    'trainset: 1 0:1 ${TRAIN_DATA_PATH}_text_document'\
    'validset: 1 0:1  ${VALIDSET_DATA_PATH}_text_document'\
    --no-shuffle \
    --new-dataset \
    --eval-first \
    --load $CHECKPOINT_PATH \
    --data-impl mmap \
    --distributed-backend nccl \
     $DEEPSPEED_ARGS \
    "

echo $CMD

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

clear; srun --jobid $SLURM_JOBID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "EXP INDEX: $SLURM_ARRAY_TASK_ID"
head -1 $SYNTHETIC_TRAIN_DATA_PATH
echo "END TIME: $(date)"
