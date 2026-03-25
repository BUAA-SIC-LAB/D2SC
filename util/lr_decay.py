
def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Apply learning rate decay based on the network layers, with smaller learning
    rates near the input and larger learning rates near the output.
    """

    param_group_names = {}
    param_groups = {}

    encoder_depth = len(model.blocks) if hasattr(model, 'blocks') else 0
    decoder_depth = len(model.decoder_blocks) if hasattr(model, 'decoder_blocks') else 0
    num_layers = encoder_depth + decoder_depth + 1  # +1 表示头部层

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers, encoder_depth)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers, encoder_depth):

    if name in ['cls_token', 'pos_embed'] or name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('decoder_blocks'):
        return encoder_depth + int(name.split('.')[1]) + 1
    else:
        return num_layers
