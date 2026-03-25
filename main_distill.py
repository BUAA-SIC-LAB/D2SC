import argparse
import datetime
import json
import numpy as np
import os
from util.dataset_distill import load_data
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_distill import train_one_epoch
from computional_demand import profile_all_static


def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0", "False"}
    TRUTHY_STRINGS = {"on", "true", "1", "True"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    elif s in {True, False}:
        return s
    else:
        print(s)
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE for pre-training distillation within federated learning architecture', add_help=False)

    # 模型参数
    parser.add_argument('--teacher_model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--student_model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask', default=True, type=bool_flag, help="use mask")
    parser.add_argument('--pixel_loss', default=False, type=bool_flag, help="use mask")
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--drop_path', default=0., type=float,
                        help='student drop path')
    parser.add_argument('--clip_grad', default=3., type=float,
                        help='student drop path')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # 联邦参数
    parser.add_argument('--n_clients', default=4, type=int,
                        help='number of the clients')
    parser.add_argument('--num_local_epochs', default=2, type=int,
                        help='number of training epoachs for one client')
    parser.add_argument('--NIID', default=True, type=bool_flag, help="use mask")
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Parameter for NIID dataset partition')

    # 蒸馏参数
    parser.add_argument('--teacher_path', default='mae_base-1599.pth', type=str,
                        help='teacher checkpoint')
    parser.add_argument('--distill_loss', default="l1", type=str, help="distillation loss type, l1, kl")
    parser.add_argument('--T', default=2, type=int, help="parameter for distillation")
    parser.add_argument('--loss_weight', default=0., type=float, help='distillation weight between two loss')
    parser.add_argument('--target_prenorm', default=False, type=bool_flag, help="use mask")
    parser.add_argument('--target_patchem', default=False, type=bool_flag, help="use mask")
    parser.add_argument('--qkv_relation_loss', default=False, type=bool_flag, help="use mask")
    parser.add_argument('--qkv_relation_weight', default=0.1, type=float, help="use mask")
    parser.add_argument('--last_layer_weight', default=0., type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--layer_index', default=4, type=int,
                        help='decoder_layer_index')
    parser.add_argument('--supervise_all', default=True, type=bool_flag, help="use mask")

    # 优化参数
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--beta', type=float, default=0.95, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=2.5e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # 数据集参数
    parser.add_argument('--data_path', type=str, help='dataset path')
    parser.add_argument('--data_type', default='cifar10', type=str,
                        help='dataset type')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=bool_flag, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # 训练参数
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='LOG_DIR',
                        help='path where to tensorboard log')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    return parser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    client_loaders = load_data(args)

    args.log_dir = os.path.join(args.output_dir, args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    loss_scaler = NativeScaler()
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # "suplayer" indicates from which layer the teacher model is distilled.
    model_teacher = models_mae.__dict__[args.teacher_model](norm_pix_loss=args.norm_pix_loss, teacher=True,
                                                            suplayer=args.layer_index, supervise_all=args.supervise_all)
    model = models_mae.__dict__[args.student_model](norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                                    encoder_pred_channel=model_teacher.embed_dim,
                                                    decoder_pred_channel=model_teacher.decoder_embed_dim,
                                                    supervise_all=args.supervise_all)
    checkpoint = torch.load(args.teacher_path, map_location='cpu')
    print('------------------------load teacher-------------------')
    print(args.teacher_path)
    missing_keys, unexpected_keys = model_teacher.load_state_dict(checkpoint['model'], strict=False)
    print('missing_keys:', missing_keys)
    print('unexpected_keys', unexpected_keys)
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.to(device)
    model_teacher.eval()
    model.to(device)

    profile_results = profile_all_static(model_teacher, model, args, device)
    print(profile_results)

    if args.resume:
        misc.load_model_fl(args=args, model=model)
    else:
        print("Training will start from the beginning.")
    print("Student Model = %s" % str(model))

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        model_diff = train_one_epoch(
            model, client_loaders, device, epoch, loss_scaler, log_writer=log_writer,
            args=args, model_teacher=model_teacher, num_local_epochs=args.num_local_epochs
        )
        is_checkpoint_epoch = ((epoch + 1) % 5 == 0) or (epoch + 1 == args.epochs)
        if args.output_dir and is_checkpoint_epoch:
            misc.save_model_fl(args=args, model=model, epoch=epoch)

        log_stats = {
            "epoch": epoch,
            "num_clients": args.n_clients,
            "model_update": model_diff,
            "NIID": args.NIID,
            "alpha": args.alpha,
            "mask_ratio": args.mask_ratio,
            "student_model": args.student_model
        }

        if args.output_dir:
            if log_writer is not None:
                print(f"Epoch: {epoch}; model_diff: {model_diff}")
                log_writer.add_scalars('model_diff', {'diff': model_diff}, epoch)
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(args.output_dir, "pretrain", timestamp)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
