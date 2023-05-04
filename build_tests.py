from model import resnet_backbone, fpn

def main():
    resnet_backbone.run_tests()
    fpn.run_tests()

if __name__ == '__main__':
    main()