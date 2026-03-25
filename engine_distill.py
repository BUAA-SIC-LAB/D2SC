import math
import sys
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn as nn
import torch.nn.functional as F
import copy
import timm.optim.optim_factory as optim_factory


def train_one_epoch(global_model, client_loaders, device,  epoch, loss_scaler,
                    args, log_writer=None, model_teacher=None, num_local_epochs=1):

    global_model.train()
    client_states = {}

    for cid, loader in client_loaders.items():

        # Create local models and build new optimizers for them.
        local_model = copy.deepcopy(global_model).to(device).train()

        param_groups = optim_factory.add_weight_decay(local_model, args.weight_decay)
        local_optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, args.beta))

        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        print(f"\n=======Epoch [{epoch}]  Client [{cid}]  LocalEpochs [{num_local_epochs}]=======\n")

        for local_epoch in range(num_local_epochs):
            train_for_client(local_model, loader, local_optimizer, device, local_epoch, loss_scaler,
                             log_writer, args, model_teacher, num_local_epochs, epoch, cid)

        client_states[cid] = copy.deepcopy(local_model.state_dict())
        del local_model, local_optimizer
        torch.cuda.empty_cache()
        print(f"Client [{cid}] finished: Samples={len(loader.dataset)}")

    return aggregate_update(global_model, client_states, client_loaders, device)


def train_for_client(local_model, data_loader, optimizer, device, local_epoch, loss_scaler,
                     log_writer, args, model_teacher, num_local_epochs, global_epoch, cid):

    local_model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f"LocalEpoch [{local_epoch}] / [{num_local_epochs}] for {cid}"
    print_freq = 20
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            progress = global_epoch + (local_epoch + data_iter_step / len(data_loader)) / num_local_epochs
            lr_sched.adjust_learning_rate(optimizer, progress, args)

        samples = samples.to(device)

        # Use AMP during the forward process
        with torch.cuda.amp.autocast():
            num_patches = local_model.patch_embed.num_patches
            noise = torch.rand(samples.shape[0], num_patches, device=samples.device)

            with torch.no_grad():
                (pred_decoder_feature_tea, decoder_hidden_tea, mask, prenorm_feature_teacher,
                 qkv_list_encoder_tea, qkv_list_decoder_tea) = model_teacher(samples, noise, mask_ratio=args.mask_ratio)

            (pred_decoder_feature_stu, decoder_hidden_stu, mask, prenorm_feature,
             pred_feature, qkv_list_encoder_stu, qkv_list_decoder_stu) = local_model(samples, noise, mask_ratio=args.mask_ratio)
            # The loss must be computed inside torch.cuda.amp.autocast, otherwise the loss may not decrease.
            (loss, loss_encoder_value,
             loss_decoder_value) = distillation_loss(args, pred_decoder_feature_stu, decoder_hidden_tea,
                                                     pred_feature, prenorm_feature_teacher)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        update_grad = ((data_iter_step + 1) % accum_iter == 0) or (data_iter_step + 1 == len(data_loader))
        loss_scaler(loss, optimizer, parameters=local_model.parameters(), update_grad=update_grad)
        if update_grad:
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_encoder_value=loss_encoder_value)
        metric_logger.update(loss_decoder_value=loss_decoder_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None and update_grad:
            step_1000x = (global_epoch * num_local_epochs * len(data_loader) +
                          local_epoch * len(data_loader) + data_iter_step)
            log_writer.add_scalars('train_loss', {f'client_{cid}': loss_value}, step_1000x)
            log_writer.add_scalars('lr', {f'client_{cid}': lr}, step_1000x)
            log_writer.flush()


def distillation_loss(args, decoder_feature_stu, decoder_feature_tea, encoder_feature_stu, encoder_feature_tea):
    loss_type = args.distill_loss.lower()
    if loss_type == "kl":
        # decoder loss
        loss_feature_decoder = F.kl_div(F.log_softmax(decoder_feature_stu / args.T, dim=-1),
                                        F.softmax(decoder_feature_tea.detach() / args.T, dim=-1),
                                        reduction='batchmean') * (args.T ** 2)
        # encoder loss
        loss_feature = F.kl_div(F.log_softmax(encoder_feature_stu / args.T, dim=-1),
                                F.softmax(encoder_feature_tea.detach() / args.T, dim=-1),
                                reduction='batchmean') * (args.T ** 2)
    # L1 loss
    else:
        labels_feature_decoder = nn.LayerNorm(decoder_feature_tea.shape[-1], eps=1e-6,
                                              elementwise_affine=False)(decoder_feature_tea)
        loss_feature_decoder = F.smooth_l1_loss(decoder_feature_stu, labels_feature_decoder, beta=2.0)
        labels_feature = nn.LayerNorm(encoder_feature_tea.shape[-1], eps=1e-6,
                                      elementwise_affine=False)(encoder_feature_tea)
        loss_feature = F.smooth_l1_loss(encoder_feature_stu, labels_feature, beta=2.0)

    loss = args.loss_weight * loss_feature + loss_feature_decoder
    loss_encoder_value = loss_feature.item()
    loss_decoder_value = loss_feature_decoder.item()
    return loss, loss_encoder_value, loss_decoder_value


def aggregate_update(global_model, client_states, client_loaders, device):

    # aggregate global model on cpu
    cpu = torch.device("cpu")

    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    avg_state = {k: torch.zeros_like(v, device=cpu) for k, v in global_state.items()}
    total_samples = sum(len(loader.dataset) for loader in client_loaders.values())

    for cid, state in client_states.items():
        client_weight = len(client_loaders[cid].dataset) / total_samples
        for k, v in state.items():
            avg_state[k] += v.cpu() * client_weight

    global_model.to(cpu).load_state_dict(avg_state)

    model_diff = 0.0
    for key in global_state:
        diff = global_state[key] - avg_state[key]
        model_diff += torch.norm(diff, p=2).item() ** 2
    model_diff = model_diff ** 0.5

    global_model.to(device)

    del client_states, avg_state, global_state
    return model_diff
