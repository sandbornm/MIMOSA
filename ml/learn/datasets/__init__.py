from torchvision import transforms

from . import BytesDatasets
from . import ImageDatasets


def build_dataset(args):
    transform = transforms.Compose([transforms.ToTensor()])
    modality = args['modality'].lower()
    if modality == 'image':
        dataset = ImageDatasets.MalwareImageDataset(args)
    elif modality == 'bytes':
        dataset = BytesDatasets.MalwareBytesDataset(args)
    else:
        raise ValueError('Unknown modality: ', args['modality'])

    return dataset
