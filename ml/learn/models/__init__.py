from . import ImageModels
from . import BytesModels


def build_model(args, n_classes):
    modality = args.modality.lower()
    arch = args.arch.lower()
    if modality.lower() == 'image':
        if 'resnext' in arch.lower():
            net = ImageModels.ResNeXt50MultilabelClassifier(arch.lower(), n_classes, pretrained=args.pretrain, variant=args.variant)
        elif 'resnet' in arch.lower():
            net = ImageModels.ResNet34MultilabelClassifier(arch.lower(), n_classes, pretrained=args.pretrain, variant=args.variant)
        elif 'convnext' in arch.lower():
            net = ImageModels.ConvNeXtMultilabelClassifier(arch.lower(), n_classes, pretrained=args.pretrain, variant=args.variant)
        else:
            raise ValueError('Unknown architecture: ', args.arch)
        criterion = ImageModels.criterion
    elif modality.lower() == 'bytes':
        if 'conv' in arch.lower():
            net = BytesModels.Conv1DBytesMultilabelClassifier(arch.lower(), n_classes, n_hidden=args.hidden, variant=args.variant)
        elif 'ff' in arch.lower():
            net = BytesModels.FFBytesMultilabelClassifier(arch.lower(), args.size, n_classes, n_hidden=args.hidden, variant=args.variant)
        else:
            raise ValueError('Unknown architecture: ', args.arch)
        criterion = BytesModels.criterion
    else:
        raise ValueError('Unknown modality: ', args.modality)

    return net, criterion
