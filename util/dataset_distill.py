from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from collections import defaultdict
import torchvision.datasets as datasets
import os
import matplotlib.pyplot as plt


def iid_partition(y, n_clients):
    """
    Divide the dataset evenly in a random way, where y represents the list of labels.
    """
    idxs = np.random.permutation(len(y))
    splits = np.array_split(idxs, n_clients)
    return {cid: split for cid, split in enumerate(splits)}


def niid_dirichlet_partition(y, n_clients, alpha, n_classes=10, min_size=10):
    """
    Split the dataset by label y into a non-IID distribution that follows a Dirichlet distribution.

    net_dataidx_map: {client_id: np.array(indices)}
    cls_counts: {client_id: {class: count}}
    """
    while True:
        idx_batch = [[] for _ in range(n_clients)]

        for k in range(n_classes):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)

            props = np.random.dirichlet([alpha] * n_clients)
            props = props / props.sum()
            cut_points = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
            idx_split = np.split(idx_k, cut_points)
            for client_id, idx in enumerate(idx_split):
                idx_batch[client_id].extend(idx)

        if min(len(idxs) for idxs in idx_batch) >= min_size:
            break

    net_dataidx_map = {cid: np.asarray(idxs, dtype=np.int64) for cid, idxs in enumerate(idx_batch)}

    cls_counts = defaultdict(dict)
    for cid, idxs in net_dataidx_map.items():
        uniq, cnt = np.unique(y[idxs], return_counts=True)
        cls_counts[cid] = {int(k): int(v) for k, v in zip(uniq, cnt)}

    return net_dataidx_map, cls_counts


def load_data(args):
    dataset_name = args.data_type.lower()
    if dataset_name.startswith("imagenet"):
        if dataset_name == "imagenet-tiny":
            random_crop = transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC)
        else:
            random_crop = transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0),
                                                       interpolation=transforms.InterpolationMode.BICUBIC)
        train_tf = transforms.Compose([
            random_crop,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        train_set = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=train_tf)
    else:
        train_tf = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010]),
        ])

        DatasetCls = CIFAR100 if dataset_name == "cifar100" else CIFAR10
        train_set = DatasetCls(args.data_path, train=True, transform=train_tf, download=True)

    nb_classes = len(train_set.classes)
    y_train = np.array(train_set.targets)
    if args.NIID:
        distribution_map, cls_counts = niid_dirichlet_partition(y_train, args.n_clients,
                                                                args.alpha, nb_classes)
    else:
        distribution_map = iid_partition(y_train, args.n_clients)

    # data distribution visualization
    # distribution_visualization(distribution_map, y_train, nb_classes, args.n_clients, args.alpha)

    client_loaders = {
        cid: DataLoader(Subset(train_set, idxs), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_mem)
        for cid, idxs in distribution_map.items()
    }
    return client_loaders


def distribution_visualization(distribution_map, y_train, nb_classes, n_clients, beta):
    mat = np.zeros((n_clients, nb_classes), dtype=int)
    for cid, idxs in distribution_map.items():
        labels = y_train[idxs]
        hist = np.bincount(labels, minlength=nb_classes)
        mat[cid] = hist
        print(f"Client {cid:2d}: {hist.tolist()}")

    scale = 0.5
    xs, ys, areas = [], [], []
    for cid in range(n_clients):
        for cls in range(nb_classes):
            xs.append(cid + 1)
            ys.append(cls + 1)
            areas.append(mat[cid, cls] * scale)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, s=areas, color="red")

    ax.set_title(rf"Data Distribution ($\beta={beta}$)", fontsize=14, pad=10)
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Class ID")
    ax.set_xticks(range(1, nb_classes + 1))
    ax.set_yticks(range(1, n_clients + 1))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.show()


# def get_args():
#     parser = argparse.ArgumentParser('pre-training distillation', add_help=False)
#     parser.add_argument("--data_type", default="cifar100", choices=["cifar10", "cifar100", "imagenet"])
#     parser.add_argument("--data_path", default="./dataset")
#     parser.add_argument("--input_size", type=int, default=32)
#     parser.add_argument("--batch_size", type=int, default=256)
#     parser.add_argument("--num_workers", type=int, default=2)
#     parser.add_argument("--pin_mem", action="store_true")
#     parser.add_argument("--n_clients", type=int, default=10)
#     parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet α")
#     parser.add_argument("--NIID", default=True, action="store_true")
#     return parser
#
#
# if __name__ == '__main__':
#     args = get_args()
#     args = args.parse_args()
#     client_loaders = load_data(args)
