from . import ImageModels
from . import BytesModels


def build_model(args, n_classes):
    modality = args.modality.lower()
    arch = args.arch.lower()
    if modality == 'image':
        if arch == 'resnext':
            net = ImageModels.ResNeXt50MultilabelClassifier(n_classes, pretrained=args.pretrain)
        elif arch == 'resnet':
            net = ImageModels.ResNet34MultilabelClassifier(n_classes, pretrained=args.pretrain)
        else:
            raise ValueError('Unknown architecture: ', args.arch)
        criterion = ImageModels.criterion
    elif modality == 'bytes':
        if arch == 'conv':
            net = BytesModels.Conv1DBytesMultilabelClassifier(n_classes, n_hidden=args.hidden, variant=args.variant)
        elif arch == 'ff':
            net = BytesModels.FFBytesMultilabelClassifier(args.size, n_classes, n_hidden=args.hidden, variant=args.variant)
        else:
            raise ValueError('Unknown architecture: ', args.arch)
        criterion = BytesModels.criterion
    else:
        raise ValueError('Unknown modality: ', args.modality)

    return net, criterion
