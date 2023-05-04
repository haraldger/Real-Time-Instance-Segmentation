from model import resnet_backbone,protonet

def main():
    resnet_backbone.run_tests()
    protonet.run_tests()

if __name__ == '__main__':
    main()