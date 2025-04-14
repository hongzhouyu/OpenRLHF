set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path /root/model/qwen-dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain /root/model/qwen \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset /root/dataset \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)
    # --flash_attn

# chmod +x /root/OpenRLHF/my_tools/train_dpo.sh

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
