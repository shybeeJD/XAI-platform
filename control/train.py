import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, mnist
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from ex_methods.module.model_mnist import Network_mnist
from ex_methods.module.model_cifar10 import Network_cifar10


def Train(model_name, **kwargs):
    # 训练超参数
    train_batchsize = 64  # 训练批大小
    test_batchsize = 128  # 测试批大小
    num_epoches = 20  # 训练轮次
    lr = kwargs.get("learning_rate")  # 学习率
    momentum = 0.5  # 动量参数，用于优化算法

    # 定义数据转换对象
    '''
    前者将数据放入tensor中，后者是归一化处理，
    两个0.5分别表示对张量进行归一化的全局平均值和方差。因图像是灰色的只有一个通道，如果有多个通道，需要有多个数字，如三个通道，应该是Normalize([m1,m2,m3], [n1,n2,n3])
    '''

    #获取mnist训练数据
    dataset = kwargs.get("dataset")
    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
        train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=True)
        test_dataset = mnist.MNIST('./data', train=False, transform=transform, download=False)
        model = Network_mnist(kwargs)
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        train_dataset = CIFAR10('./data', train=True, transform=transform, download=True)
        test_dataset = CIFAR10('./data', train=False, transform=transform, download=False)
        model = Network_cifar10(kwargs)

    #datalodar用于加载训练数据
    train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batchsize,shuffle=False)

    # 判断当前设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model.to(device)
    # print(model)

    # 损失函数
    loss_func = kwargs.get("loss_func")
    if loss_func == "ESM":
        criterion = nn.MSELoss()
    elif loss_func == "Cross-entropy":
        criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = kwargs.get("optimizer")
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # 统计损失值和精确度
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        model.train()

        # 读取数据
        for img, label in train_loader:
            # 将数据放入设备中
            img = img.to(device)
            label = label.to(device)

            # 向模型中输入数据
            out = model.forward(img)
            # 计算损失值
            loss = criterion(out, label)
            # 清理当前优化器中梯度信息
            optimizer.zero_grad()
            # 根据损失值计算梯度
            loss.backward()
            # 根据梯度信息进行模型优化
            optimizer.step()

            # 统计损失信息
            train_loss += loss.item()

            # 得到预测值
            _, pred = out.max(1)

            # 判断预测正确个数，计算精度
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc

        losses.append(train_loss/len(train_loader))
        acces.append((train_acc/len(train_loader)))

        # 进行模型评估
        eval_loss = 0
        eval_acc = 0
        model.eval()

        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)

            out = model.forward(img)
            loss = criterion(out, label)

            # 记录误差
            eval_loss += loss.item()

            # 记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc

        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))

        # 打印学习情况
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
              .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                      eval_loss / len(test_loader), eval_acc / len(test_loader)))

    model_detail = {
        "model_name": model_name,
        "model": model,
        "parameters": kwargs
    }
    torch.save(model_detail, "models/" + model_name +".pkl")
    return {"train_acc": train_acc / len(train_loader),
            "eval_acc": eval_acc / len(test_loader)}
