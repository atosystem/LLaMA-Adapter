# specify GPUS
export CUDA_VISIBLE_DEVICES=4,5,6


torchrun --nproc_per_node 3 finetuning.py \
         --model Llama7B_adapter \
         --llama_model_path /data1/scratch/yjshih/pretrained_models/LLaMA-7B/ \
         --data_path /data1/scratch/yjshih/data/alpaca_data.json \
         --adapter_layer 30 \
         --adapter_len 10 \
         --max_seq_len 512 \
         --batch_size 7 \
         --epochs 5 \
         --warmup_epochs 2 \
         --blr 9e-3 \
         --weight_decay 0.02 \
         --output_dir ./checkpoint/

# torchrun --nproc_per_node 8 finetuning.py \
#     --model Llama7B_adapter \
#     --llama_model_path $TARGET_FOLDER/ \
#     --data_path $DATA_PATH/alpaca_data.json \
#     --adapter_layer 30 \
#     --adapter_len 10 \
#     --max_seq_len 512 \
#     --batch_size 4 \
#     --epochs 5 \
#     --warmup_epochs 2 \
#     --blr 9e-3 \
#     --weight_decay 0.02 \
#     --output_dir ./checkpoint/
