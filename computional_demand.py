import copy
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


def model_params_m(model, trainable_only=False):
    if trainable_only:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())

    return {
        "params": total_params,
        "params_m": total_params / 1e6,
    }


def build_distill_inputs(model, batch_size, input_size, device):
    samples = torch.randn(batch_size, 3, input_size, input_size, device=device)
    num_patches = model.patch_embed.num_patches
    noise = torch.rand(batch_size, num_patches, device=device)
    return samples, noise


class DistillMACWrapper(nn.Module):

    def __init__(self, model, input_size, mask_ratio=0.75):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.mask_ratio = mask_ratio

    def forward(self, x):
        num_patches = self.model.patch_embed.num_patches
        noise = torch.rand(x.shape[0], num_patches, device=x.device)

        out = self.model(x, noise, mask_ratio=self.mask_ratio)

        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)):
            for v in out:
                if torch.is_tensor(v):
                    return v

        raise RuntimeError("no tensor output")


def count_macs(model, input_size, mask_ratio, name="model"):
    """
    Count the MACs for a single forward pass.
    """
    wrapped = DistillMACWrapper(
        copy.deepcopy(model).cpu().eval(), input_size=input_size, mask_ratio=mask_ratio)
    macs, _ = get_model_complexity_info(
        wrapped, (3, input_size, input_size),
        as_strings=False, print_per_layer_stat=False, verbose=False)

    return {
        "name": name,
        "macs": macs,
        "macs_g": macs / 1e9,
    }


@torch.no_grad()
def profile_model_infer_memory(model, args, device, batch_size=None, use_amp=True):
    if device.type != "cuda":
        return None

    bs = batch_size if batch_size is not None else args.batch_size

    test_model = copy.deepcopy(model).to(device).eval()

    samples = torch.randn(bs, 3, args.input_size, args.input_size, device=device)
    num_patches = test_model.patch_embed.num_patches
    noise = torch.rand(bs, num_patches, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    if use_amp:
        with torch.cuda.amp.autocast():
            _ = test_model(samples, noise, mask_ratio=args.mask_ratio)
    else:
        _ = test_model(samples, noise, mask_ratio=args.mask_ratio)

    torch.cuda.synchronize(device)

    result = {
        "batch_size": bs,
        "peak_alloc_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }

    del test_model, samples, noise
    torch.cuda.empty_cache()
    return result


def reset_cuda_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def read_cuda_peak_mb(device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024 ** 2,
    }


def profile_all_static(model_teacher, model_student, args, device):
    """
    Measure the computational cost of the first-stage model.
    """
    teacher_size = model_params_m(model_teacher)
    student_size = model_params_m(model_student)

    teacher_macs = count_macs(model_teacher, args.input_size, args.mask_ratio, name="teacher")
    student_macs = count_macs(model_student, args.input_size, args.mask_ratio, name="student")

    teacher_mem = profile_model_infer_memory(model_teacher, args, device, batch_size=args.batch_size)
    student_mem = profile_model_infer_memory(model_student, args, device, batch_size=args.batch_size)

    return {
        "teacher_param": teacher_size,
        "student_param": student_size,
        "teacher_macs": teacher_macs,
        "student_macs": student_macs,
        "teacher_infer_mem": teacher_mem,
        "student_train_mem": student_mem,
    }


def count_macs_single(model, args, name="model"):
    model_cpu = copy.deepcopy(model).cpu().eval()

    for m in model_cpu.modules():
        if m.__class__.__name__ == "Attention" and hasattr(m, "num_heads") and not hasattr(m, "head_dim"):
            if hasattr(m, "qkv") and hasattr(m.qkv, "in_features"):
                m.head_dim = m.qkv.in_features // m.num_heads

    macs, _ = get_model_complexity_info(model_cpu, (3, args.input_size, args.input_size),
                                        as_strings=False, print_per_layer_stat=False, verbose=False)
    return {
        "name": name,
        "macs": macs,
        "macs_g": macs / 1e9,
    }


@torch.no_grad()
def profile_infer_memory(model, args, device, batch_size=1):
    """
    Measure the GPU memory cost of the second-stage model.
    """
    if device.type != "cuda":
        return None

    model = copy.deepcopy(model).to(device).eval()
    x = torch.randn(batch_size, 3, args.input_size, args.input_size, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    _ = model(x)
    torch.cuda.synchronize(device)
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024 ** 2,
    }


def profile_finetune_static(model, args, device):
    """
    Measure the computational cost of the second-stage model.
    """
    return {
        "macs": count_macs_single(model, args, name=args.model),
        "infer_mem": profile_infer_memory(model, args, device, batch_size=1),
    }
