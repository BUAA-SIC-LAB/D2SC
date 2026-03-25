IMAGENET_DIR='/home/serika/CIFAR_torch'
JOB_DIR='output_dir'
python main_distill.py \
--data_path $IMAGENET_DIR --batch_size 256  --accum_iter 2 \
--teacher_model mae_vit_base_patch16 --loss_weight 1 \
--student_model mae_vit_tiny_patch16_dec128d4b \
--mask_ratio 0.75 --mask True --norm_pix_loss  --distill_loss l1 \
--epochs 100 --blr 1.5e-4 --weight_decay 0.05 --warmup_epochs 10 \
--n_clients 10 --num_local_epochs 2 --NIID True --alpha 0.6 --data_type cifar10 \
--output_dir $JOB_DIR --log_dir LOG_DIR --drop_path 0.0  --beta 0.95   \
--teacher_path  'Teacher/mae_visualize_vit_base.pth'