from . import ImageModels
from . import BytesModels


def build_model(args, n_classes):
    modality = args['modality'].lower()
    arch = args['arch'].lower()
    pretrain = args['pretrain']
    variant = args['variant']
    if modality.lower() == 'image':
        if 'resnext' in arch.lower():
            net = ImageModels.ResNeXt50MultilabelClassifier(arch.lower(), n_classes, pretrained=pretrain, variant=variant)
        elif 'resnet' in arch.lower():
            net = ImageModels.ResNet34MultilabelClassifier(arch.lower(), n_classes, pretrained=pretrain, variant=variant)
        elif 'convnext' in arch.lower():
            net = ImageModels.ConvNeXtMultilabelClassifier(arch.lower(), n_classes, pretrained=pretrain, variant=variant)
        else:
            raise ValueError('Unknown architecture: ', args['arch'])
    elif modality.lower() == 'bytes':
        if 'conv' in arch.lower():
            net = BytesModels.Conv1DBytesMultilabelClassifier(arch.lower(), n_classes, n_hidden=args['hidden'], variant=variant)
        elif 'ff' in arch.lower():
            net = BytesModels.FFBytesMultilabelClassifier(arch.lower(), args['size'], n_classes, n_hidden=args['hidden'], variant=variant)
        else:
            raise ValueError('Unknown architecture: ', args['arch'])
    else:
        raise ValueError('Unknown modality: ', args['modality'])

    return net
