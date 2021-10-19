import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os
from PIL import Image


def sort_files(files):
    for file in files:
        if f"adv_lrp" in file:
            adv_lrp = file
        elif f"adv_heatmap" in file:
            adv_h = file
        elif f"adv_lime" in file:
            adv_lime = file
        elif f"nor_lrp" in file:
            nor_lrp = file
        elif f"nor_heatmap" in file:
            nor_heatmap = file
        elif f"nor_lime" in file:
            nor_lime = file
    return [nor_lrp, nor_heatmap, nor_lime, adv_lrp, adv_h, adv_lime]


def plot_overview(images_per_row=3):
    """
    Helper method for plotting the result of the attack
    """
    captions = ['normal_lrp', 'normal_heatmap', 'normal_lime', 'adversarial_lrp', 'adversarial_heatmap',
                'adversarial_lime']
    path = "web/static/images/result"
    filename = 'web/static/images/overview/overview.png'
    files = os.listdir(path)
    s_files = sort_files(files)
    imgs = []
    for file in s_files:
        file_path = os.path.join(path, file)
        img = Image.open(file_path)
        imgs.append(img)
    plot_grid(imgs, captions, filename=filename, images_per_row=images_per_row)
    return filename.split("static/")[1]


def plot_grid(images, titles=None, images_per_row=3, cmap='gray', norm=mpl.colors.NoNorm(), filename="overview.png"):
    """
    Helper method to plot a grid with matplotlib
    """
    plt.close("all")
    num_images = len(images)
    images_per_row = min(num_images, images_per_row)

    num_rows = math.ceil(num_images / images_per_row)

    if len(cmap) != num_images or type(cmap) == str:
        cmap = [cmap] * num_images

    fig, axes = plt.subplots(nrows=num_rows, ncols=images_per_row)

    fig = plt.gcf()
    fig.set_size_inches(4 * images_per_row, 5 * int(np.ceil(len(images) / images_per_row)))
    for i in range(num_rows):
        for j in range(images_per_row):

            idx = images_per_row * i + j

            if num_rows == 1:
                a_ij = axes[j]
            elif images_per_row == 1:
                a_ij = axes[i]
            else:
                a_ij = axes[i, j]
            a_ij.axis('off')
            if idx >= num_images:
                break
            a_ij.imshow(images[idx], cmap=cmap[idx], norm=norm, interpolation='nearest')
            a_ij.set_title(titles[idx])

    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)

    plt.savefig(filename)
    plt.close()
