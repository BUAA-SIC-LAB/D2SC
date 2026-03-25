import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
import util.lr_decay as lrd
import util.misc as misc
from util.dataset_finetune import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit_task
from engine_finetune_task import train_epoch_for_task, evaluate_task
from timm.data.mixup import Mixup
from computional_demand import profile_finetune_static


def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def parse_snr(value):
    return list(map(int, value.split(',')))


def get_args_parser():
    parser = argparse.ArgumentParser('ViT finetune for image classification or reconstruction', add_help=False)

    # 模型参数
    parser.add_argument('--model', default='vit_tiny_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--interpolate_position', default=False, type=bool_flag,
                        help='Interpolate position token or not')

    # 通信参数
    parser.add_argument('--channel_type', default='rayleigh', type=str,
                        help='The channel type for semantic communication, you can choose awgn, rayleigh or none')
    parser.add_argument('--snr_set', type=parse_snr, default='1,4,7,10,13',
                        help='random snr set')
    parser.add_argument('--given_snr', type=int, default=None,
                        help='The given SNR for semantic communication')
    parser.add_argument('--modulation_layers', type=int, default=3,
                        help='The modulation layers for SNRModulation')
    parser.add_argument('--pass_channel', type=bool_flag, default=True)
    parser.add_argument('--distortion_metric', type=str, default='MSE',
                        help='The loss function for reconstruction, you can choose MSE, MS-SSIM')

    # 训练参数
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='LOG_DIR',
                        help='path where to tensorboard log')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--loss_weight', type=float, default=0.01,
                        help='Loss weight for information bottleneck')

    # 优化参数
    parser.add_argument('--clip_grad', type=float, default=2., metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # 图像增强参数
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    # 随机擦除，图像增强的手段
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # 数据集参数
    parser.add_argument('--data_path', type=str, help='dataset path')
    parser.add_argument('--data_type', default='cifar10', type=str,
                        help='dataset type, you can choose cifar10, cifar100, imagenet, imagenet-tiny')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser.parse_args()


def evaluate_model(args, data_loader_val, model, device):
    test_stats = evaluate_task(data_loader_val, model, device)
    print(f"Accuracy 1 on the {args.data_type}: {test_stats['acc1']:.2f}%")
    print(f"Accuracy 5 on the {args.data_type}: {test_stats['acc5']:.2f}%")
    return test_stats


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    dataset_train, dataset_val = build_dataset(args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    args.log_dir = os.path.join(args.output_dir, args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit_task.__dict__[args.model](
        args=args,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
                                        no_weight_decay_list=model.no_weight_decay(),
                                        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    model.to(device)

    if args.resume:
        misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
    elif args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        # If there is a classification head, it means the task is a classification task. If the number of classification
        # heads in the original model and the checkpoint model is different, remove the classification head.
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if args.interpolate_position:
            interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        print("------------------------Load model------------------------")
        # Initialize the weights of the model classification head with a truncated normal distribution, and discard
        # the original classification head.
        trunc_normal_(model.head.weight, std=2e-5)

    print("Model = %s" % str(model))
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    profile_results = profile_finetune_static(model, args, device)
    print(profile_results)

    if args.eval:
        _ = evaluate_model(args, data_loader_val, model, device)
        exit(0)

    max_accuracy = 0.0

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_epoch_for_task(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer, args=args, mixup_fn=mixup_fn
        )

        if args.output_dir:
            if epoch % 5 == 0:
                misc.save_model(args=args, model=model, epoch=epoch,
                                loss_scaler=loss_scaler, optimizer=optimizer)
            elif epoch + 1 == args.epochs:
                misc.save_model(args=args, model=model, epoch=epoch)

        test_stats = evaluate_model(args, data_loader_val, model, device)
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        if log_writer is not None:
            log_writer.add_scalar('task/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('task/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('task/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    if args.output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(args.output_dir, "finetune", timestamp)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
