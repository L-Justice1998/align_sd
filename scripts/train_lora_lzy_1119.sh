CUDA_VISIBLE_DEVICES=6 accelerate launch  train_text_to_image_lora_lzy.py \
--pretrained_model_name_or_path models/sd1.4 \
--regularization_annotation /data/lzy/improved_aesthetics_6.5plus1/regularization_images2.jsonl \
--resolution 512 \
--train_batch_size 1 \
--num_train_epochs 1 \
--checkpointing_steps 10000 \
--learning_rate 2e-5 \
--lr_scheduler constant \
--lr_warmup_steps 0 \
--seed 42 \
--validation_prompt_file validation_prompts11.20.txt \
--validation_iters 300 \
--report_to tensorboard \
--resume_from_checkpoint latest \
--dataloader_num_workers 4 \
--max_train_steps 10001 \
--output_dir training_dir/output_dir12.06.02 \
--gradient_accumulation_steps 40 \

