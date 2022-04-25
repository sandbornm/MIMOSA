from torchvision import transforms

from . import BytesDatasets
from . import ImageDatasets


def build_dataset(args):
    transform = transforms.Compose([transforms.ToTensor()])
    modality = args.modality.lower()
    if modality == 'image':
        dataset = ImageDatasets.MalwareImageDataset(args.examples_dir, args.labels_csv, transform=transform, sz=tuple(args.size))
    elif modality == 'bytes':
        dataset = BytesDatasets.MalwareBytesDataset(args.examples_dir, args.labels_csv, sz=args.size)
    else:
        raise ValueError('Unknown modality: ', args.modality)

    return dataset
