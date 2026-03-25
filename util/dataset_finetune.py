import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import CIFAR10, CIFAR100


def build_dataset(args):
    dataset_name = args.data_type.lower()
    train_transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=transforms.InterpolationMode.BICUBIC,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    if dataset_name.startswith("imagenet"):

        if dataset_name == "imagenet-tiny":
            random_crop = transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC)
        else:
            random_crop = transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0),
                                                       interpolation=transforms.InterpolationMode.BICUBIC)
        test_transform = transforms.Compose([
            random_crop,
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        train_set = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=train_transform)
        test_set = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=test_transform)
    else:

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0), ratio=(0.75, 1.33),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010]),
        ])

        DatasetCls = CIFAR100 if dataset_name == "cifar100" else CIFAR10
        train_set = DatasetCls(args.data_path, train=True, transform=train_transform, download=True)
        test_set = DatasetCls(args.data_path, train=False, transform=test_transform, download=True)

    return train_set, test_set
