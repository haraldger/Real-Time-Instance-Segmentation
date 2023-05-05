from model import resnet_backbone, fpn, prediction_heads, protonet

def main():
    resnet_backbone.run_tests()
    fpn.run_tests()
    prediction_heads.run_tests()
    protonet.run_tests()


if __name__ == '__main__':
    main()
