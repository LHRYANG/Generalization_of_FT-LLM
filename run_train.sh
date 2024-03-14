for num in 2000 4000 6000
do

deepspeed --master_port=4006 \
train.py \
--model_name_or_path llama-2-7b \
--data_path ./data/summary/xsum_train_${num}.json \
--output_dir saved_models/summary/xsum_${num} \
--report_to none \
--num_train_epochs 2 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "no" \
--save_steps 1000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0 \
--lr_scheduler_type "linear" \
--logging_steps 1 \
--deepspeed ./ds_config.json \
--tf32 False \
--bf16 True
#--fp16

done

