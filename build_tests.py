from model import resnet_backbone, fpn, prediction_heads, protonet, yolact
from utils import fast_nms
import multibox_loss

def main():
    multibox_loss.run_tests()
    fast_nms.run_tests()
    resnet_backbone.run_tests()
    fpn.run_tests()
    prediction_heads.run_tests()
    protonet.run_tests()
    yolact.run_tests()


if __name__ == '__main__':
    main()
