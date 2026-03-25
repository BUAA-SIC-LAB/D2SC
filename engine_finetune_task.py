import math
import sys
import torch
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def train_epoch_for_task(model, data_loader, optimizer, device, epoch, loss_scaler,
                         max_norm=1.0, log_writer=None, args=None, mixup_fn=None):
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            noise_std, feature, outputs = model(samples)
            cls_loss = criterion(outputs, targets)

            # If the channel is enabled, compute the mutual information loss.
            if args.pass_channel:
                gamma = (noise_std**2) / (feature**2 + 1e-8)
                log_gamma = torch.log(gamma + 1e-8)
                sigmoid_term = 0.63576 * torch.sigmoid(1.87320 + 1.48695 * log_gamma)
                log_term = 0.5 * torch.log(1 + 1 / (gamma + 1e-8))
                kl_element = -sigmoid_term + log_term + 0.63576
                kl_element = torch.clamp(kl_element, min=0, max=10.0)
                kl_per_sample = torch.sum(kl_element, dim=1)
                ib_loss = torch.mean(kl_per_sample)
                loss = cls_loss + args.loss_weight * ib_loss
            else:
                loss = cls_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                    create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_task(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            _, _, output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
