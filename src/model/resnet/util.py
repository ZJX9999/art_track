from src.model.resnet import resnet


def _make_blocks(out_channel, stride, num_block):
    """
    make blocks config
    """
    result = []
    for ch, s, num in zip(out_channel, stride, num_block):
        result.append([{
            'out_channel': ch,
            'stride': 1
        }] * (num - 1) + [{
            'out_channel': ch,
            'stride': s
        }])
    return result


def resnet_101(in_channels, **kwargs):
    """
    get resnet 101
    """
    blocks = _make_blocks([256, 512, 1024, 2048], [2, 2, 2, 1], [3, 4, 23, 3])
    net = resnet.ResNet(blocks, in_channels, **kwargs)
    return net
