# prepare verl variables
model_name=Qwen/Qwen2.5-32B
project_name='VeRL-DAPO-DAPOMath'
exp_name="official-32B-32p"
default_ckpt_path="checkpoints/${project_name}/${exp_name}"

python -m pip install transformers==4.56.0

# Set common environment variables
SERVICE_PREFIX="${LEPTON_JOB_SERVICE_PREFIX:-$LEPTON_JOB_NAME}"
SUBDOMAIN="${LEPTON_SUBDOMAIN:-$LEPTON_JOB_NAME-job-svc}"
export MASTER_ADDR=${SERVICE_PREFIX}-0.${SUBDOMAIN}
export THIS_ADDR=${SERVICE_PREFIX}-${LEPTON_JOB_WORKER_INDEX}.${SUBDOMAIN}
export WORLD_SIZE=${LEPTON_JOB_TOTAL_WORKERS}
export WORLD_RANK=${LEPTON_JOB_WORKER_INDEX}
export NGPUS=${LEPTON_RESOURCE_ACCELERATOR_NUM}

# --- 1. Install System Dependencies (including libhwloc-dev) using APT ---
apt-get update -y # Update package lists
# Install Development Tools (common equivalent for groupinstall)
apt-get install -y build-essential
# Install other dependencies
apt-get install -y \
    dkms linux-headers-$(uname -r) \
    git wget pciutils libnuma-dev libnl-3-dev libssl-dev libudev-dev \
    cmake g++ libhwloc-dev # hwloc-devel equivalent is libhwloc-dev

# --- 2. Install EFA Driver ---
# The EFA installer script should detect your OS and use 'apt' if available.
EFA_INSTALLER_VERSION=latest
wget https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz
tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz
cd aws-efa-installer
# Run EFA installer (as root, sudo is not needed)
./efa_installer.sh -y --skip-kmod --no-verify
cd ..

# Verify EFA is working before proceeding
fi_info -p efa

# --- 3. Install AWS-OFI-NCCL Plugin ---
# Clean up any previous attempt
if [ -d "aws-ofi-nccl" ]; then
  echo "Removing old aws-ofi-nccl directory..."
  rm -rf aws-ofi-nccl
fi

git clone https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
./autogen.sh

# This configure command should now find hwloc (libhwloc-dev provides hwloc.h)
# Ensure paths below are correct for your environment
./configure \
    --with-mpi=/opt/amazon/openmpi \
    --with-libfabric=/opt/amazon/efa \
    --with-cuda=/usr/local/cuda \
    --with-nccl=/usr/local/nccl \
    --prefix=/opt/aws-ofi-nccl

# Compile and install (as root, sudo is not needed for make install)
make -j $(nproc) && make install
cd ..

sleep 30

# --- 4. Set Environment Variables ---
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_PROTO="simple"
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=INFO # For NCCL debugging; set to WARN for less verbosity
# export NCCL_SOCKET_IFNAME=eth0 # Optional: Specify network interface for NCCL bootstrap

echo "Setup attempt complete. Environment variables are set for this session."


# Get master IP
NODE_IP=""
while [ -z "$NODE_IP" ]; do
    NODE_IP=$(getent hosts -- $MASTER_ADDR | awk '{ print $1 }' || echo "")
    if [ -z "$NODE_IP" ]; then
        sleep 5
    fi
done
export MASTER_IP=$NODE_IP

# Adjust environment variables
if [ ${NGPUS} != 8 ]; then
    # There are no IB devices for this resource shape, need to unset NCCL_SOCKET_IFNAME
    export NCCL_SOCKET_IFNAME=
fi

pip install "ray[train]" torch "accelerate==1.6.0" "transformers[torch]==4.51.3" datasets evaluate numpy scikit-learn pyarrow<16.2.0a0,>=16.1.0


# install verl
sleep 1
git clone -b v0.4.1 https://github.com/volcengine/verl && cd verl
sleep 1
pip3 install -e .[vllm]
pwd
ls





max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=16
train_prompt_mini_bsz=32

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

sp_size=8
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4

# preprocess data
TRAIN_FILE=data/dapo-math-17k.parquet
TEST_FILE=data/aime-2024.parquet
mkdir -p data
wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
## end data processing


# Initialize Ray cluster
if [ ${WORLD_RANK} == 0 ]; then
    ray start --head --port=6379
else
    ray start --address="${MASTER_IP}:6379" --block
fi


if [ ${WORLD_RANK} == 0 ]; then
PYTHONUNBUFFERED=1 python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${LEPTON_RESOURCE_ACCELERATOR_NUM}" \
    trainer.nnodes="${LEPTON_JOB_TOTAL_WORKERS}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=100000000 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$default_ckpt_path \
    trainer.resume_mode=auto 2>&1 | tee verl_demo_slurm.log
fi

ray stop
