import torch

NUM_CLASSES = 20
NETWORK_INPUT_SIZE = (224, 224)
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIXEL_SUPERVISION_COUNT = 5

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def get_voc_colormap():
    def bitget(number, pos):
        return (number >> pos) & 1
    colormap = torch.zeros((256, 3), dtype=torch.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << (7 - j))
            g |= (bitget(c, 1) << (7 - j))
            b |= (bitget(c, 2) << (7 - j))
            c >>= 3
        colormap[i] = torch.tensor([r, g, b], dtype=torch.uint8)
    return colormap.float() / 255.0

VOC_CMAP = get_voc_colormap()