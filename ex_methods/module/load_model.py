from ex_methods.module.model_VGG19 import VGG
from ex_methods.module.model_Resnet import resnet18, resnet34, resnet50, resnet101
from ex_methods.module.model_Densenet import densenet121
from ex_methods.module.model_mnist import mnist_self
from ex_methods.module.model_cifar10 import cifar_self


def load_model(model, pretrained=False, pretrained_path=None):
    net = None
    if pretrained_path == None:
        if model == 'VGG19':
            net = VGG().forward(pretrained=pretrained,pretrained_path=pretrained_path)
        elif model == 'Resnet18':
            net = resnet18(pretrained=pretrained,pretrained_path=pretrained_path)
        elif model == 'Resnet34':
            net = resnet34(pretrained=pretrained,pretrained_path=pretrained_path)
        elif model == 'Resnet50':
            net = resnet50(pretrained=pretrained,pretrained_path=pretrained_path)
        elif model == 'Resnet101':
            net = resnet101(pretrained=pretrained, pretrained_path=pretrained_path)
        elif model == 'Densenet121':
            net = densenet121(pretrained=pretrained, pretrained_path=pretrained_path)
        elif model == 'cifar10':
            net = cifar_self(pretrained=pretrained, pretrained_path=pretrained_path)
        elif model == 'mnist':
            net = mnist_self(pretrained=pretrained, pretrained_path=pretrained_path)
        else:
            print("model not find!")
    else:
        if 'VGG19' in pretrained_path:
            net = VGG().forward(pretrained=pretrained, pretrained_path=pretrained_path)
        elif 'Resnet18' in pretrained_path:
            net = resnet18(pretrained=pretrained, pretrained_path=pretrained_path)
        elif 'Resnet34' in pretrained_path:
            net = resnet34(pretrained=pretrained, pretrained_path=pretrained_path)
        elif 'Resnet50' in pretrained_path:
            net = resnet50(pretrained=pretrained, pretrained_path=pretrained_path)
        elif 'Resnet101' in pretrained_path:
            net = resnet101(pretrained=pretrained, pretrained_path=pretrained_path)
        elif 'Densenet121' in pretrained_path:
            net = densenet121(pretrained=pretrained,pretrained_path=pretrained_path)
        elif 'mnist' in pretrained_path:
            net = mnist_self(pretrained=pretrained, pretrained_path=pretrained_path)
        else:
            print("model not find!")
    return net
