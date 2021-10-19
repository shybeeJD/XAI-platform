from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray

from ex_methods.lime import lime_image
from ex_methods.module.load_model import load_model
from ex_methods.module.utils import grad_visualize, lrp_visualize, load_image, target_layer, preprocess_transform


def get_class_list(dataset):
    if dataset == 'Imagenet':
        with open(os.path.abspath('./data/imagenet_class_index.json'), 'r') as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            return idx2label
    elif dataset == 'mnist':
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif dataset == 'cifar10':
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        return "暂不支持其他数据集"


def get_target_num(dataset):
    if dataset == "Imagenet":
        return 1000
    else:
        return 10


def predict(img, model, device, dataset):
    net = load_model(model, pretrained=True, pretrained_path=None)
    net = net.eval().to(device)
    x = load_image(device, img, dataset)

    activation_output = net.forward(x)
    _, prediction = torch.max(activation_output, 1)
    print(prediction)

    return prediction, x, activation_output, net


def draw_grad(prediction, x, activation_output, net, dataset, model, device):
    result_lrp = net.interpretation(activation_output,
                                    interpreter="lrp",
                                    labels=prediction,
                                    num_target=get_target_num(dataset),
                                    device=device,
                                    target_layer=target_layer(model),
                                    inputs=x)
    result_cam = net.interpretation(activation_output,
                                    interpreter="grad_cam",
                                    labels=prediction,
                                    num_target=get_target_num(dataset),
                                    device=device,
                                    target_layer=target_layer(model),
                                    inputs=x)
    x = x.permute(0, 2, 3, 1).cpu().detach().numpy()
    x = x - x.min(axis=(1, 2, 3), keepdims=True)
    x = x / x.max(axis=(1, 2, 3), keepdims=True)

    img_l = lrp_visualize(result_lrp, 0.9)[0]

    img_h = grad_visualize(result_cam, x)[0]

    img_l = Image.fromarray((img_l * 255).astype(np.uint8))
    img_h = Image.fromarray((img_h * 255).astype(np.uint8))
    return img_l, img_h


def batch_predict(images, model, device, dataset):
    """
    lime 中对随机取样图像进行预测
    :param images: np.array
    :param model:
    :param device:
    :return:
    """
    if dataset == "mnist":
        images = rgb2gray(images)
    batch = torch.stack(tuple(preprocess_transform(i, dataset) for i in images), dim=0)
    batch = batch.to(device).type(dtype=torch.float32)
    probs = model.forward(batch)
    return probs.detach().cpu().numpy()


def draw_lime(img, net, device, dataset):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(img),
                                             net,
                                             device,
                                             dataset,
                                             batch_predict,  # classification function
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)  # number of images that will be sent to classification function

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, negative_only=False,
                                                num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)
    img_boundry = Image.fromarray((img_boundry * 255).astype(np.uint8))
    return img_boundry


def get_explain(nor_img, adv_img, model, dataset):
    device = torch.device("cuda:0")
    class_list = get_class_list(dataset)
    nor_image = Image.open(nor_img).convert('RGB')
    adv_image = Image.open(adv_img).convert('RGB')
    if dataset == "mnist":
        nor_image = nor_image.convert("L")
        adv_image = adv_image.convert("L")

    imgs = [nor_image, adv_image]
    class_names = []
    ex_imgs = []
    for img in imgs:
        prediction, x, activation_output, net = predict(img, model, device, dataset)
        class_name = class_list[prediction.item()]
        img_l, img_h = draw_grad(prediction, x, activation_output, net, dataset, model, device)
        img_lime = draw_lime(img, net, device, dataset)
        class_names.append(class_name)
        ex_imgs.append([img_l, img_h, img_lime])
    return class_names, ex_imgs
