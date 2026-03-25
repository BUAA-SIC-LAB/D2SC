IMAGENET_DIR='/home/serika/CIFAR_torch'
JOB_DIR='output_dir'
FINETUNE_MODEL='/home/serika/TSC/G2SC_FL/Finetune/checkpoint-99.pth'
python main_finetune.py \
--data_path $IMAGENET_DIR --batch_size 256  --accum_iter 1 \
--nb_classes 10 --loss_weight 0.001 \
--model vit_tiny_patch16 --finetune $FINETUNE_MODEL \
--epochs 100 --blr 5e-4 --weight_decay 0.05 --data_type cifar10 \
--output_dir $JOB_DIR --log_dir LOG_DIR --drop_path 0.1