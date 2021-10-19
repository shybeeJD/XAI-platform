import json
import os
from flask import request
from flask import Blueprint, render_template
from control.train import Train
from control.explain import get_explain
from control.fool import fool
from control.draw_result import plot_overview
from utils.functional import save_ex_img, clean_ex_img, save_adv_img

api = Blueprint("api", __name__, template_folder="../templates", static_folder="../static")


@api.route("/", methods=["GET", "POST"])
def home():
    return render_template("mnist.html")


@api.route("/train", methods=["POST"])
def train():
    if request.method == "POST":
        model_name = request.form.get("model_name")
        conv_k = request.form.get("conv_k")
        conv_stride = request.form.get("conv_stride")
        conv_padding = request.form.get("conv_padding")
        conv_act_func = request.form.get("conv_act_func")
        pool_k = request.form.get("pool_k")
        pool_stride = request.form.get("pool_stride")
        pool_padding = request.form.get("pool_padding")
        linear_bais = request.form.get("linear_bais")
        linear_act_func = request.form.get("linear_act_func")
        loss_func = request.form.get("loss_func")
        optimizer = request.form.get("optimizer")
        learning_rate = request.form.get("learning_rate")
        dataset = request.form.get("dataset")

        conv_k = int(conv_k[0])
        conv_stride = int(conv_stride[0])
        conv_padding = True if conv_padding[0] == "是" else False
        conv_act_func = str(conv_act_func)
        pool_stride = int(pool_stride[0])
        pool_k = int(pool_k[0])
        pool_padding = True if pool_padding[0] == "是" else False
        linear_bais = True if linear_bais[0] == "是" else False
        linear_act_func = str(linear_act_func)
        loss_func = str(loss_func)
        optimizer = str(optimizer)
        learning_rate = float(learning_rate)
        dataset = str(dataset)

        result = Train(model_name=model_name, conv_k=conv_k, conv_stride=conv_stride, conv_padding=conv_padding, conv_act_func=conv_act_func,
                       pool_stride=pool_stride, pool_k=pool_k, pool_padding=pool_padding,
                       linear_bais=linear_bais, linear_act_func=linear_act_func, loss_func=loss_func,
                       optimizer=optimizer, learning_rate=learning_rate, dataset=dataset)
        print("-> request forward", result)
        return json.dumps({"data": result})


@api.route("/explain", methods=["POST"])
def explain():
    nor_img_f = request.files.get("nor_img")
    adv_img_f = request.files.get("adv_img")
    nor_img = nor_img_f.stream
    adv_img = adv_img_f.stream

    dataset = request.form.get("dataset")
    model = request.form.get("model")

    # 正常样本
    classnames, ex_imgs = get_explain(nor_img, adv_img, model=model, dataset=dataset)
    nor_class_name, adv_class_name = classnames[0], classnames[1]
    nor_imgs, adv_imgs = ex_imgs[0], ex_imgs[1]
    nor_l_ex_img, nor_h_ex_img, nor_lime_ex_img = nor_imgs[0], nor_imgs[1], nor_imgs[2]
    adv_l_ex_img, adv_h_ex_img, adv_lime_ex_img = adv_imgs[0], adv_imgs[1], adv_imgs[2]

    clean_ex_img()
    # 存储正常样本解释图
    nor_l_f = save_ex_img(nor_l_ex_img, "nor_lrp.png")
    nor_h_f = save_ex_img(nor_h_ex_img, "nor_heatmap.png")
    nor_lime_f = save_ex_img(nor_lime_ex_img, "nor_lime.png")
    # 存储对抗样本解释图
    adv_l_f = save_ex_img(adv_l_ex_img, "adv_lrp.png")
    adv_h_f = save_ex_img(adv_h_ex_img, "adv_heatmap.png")
    adv_lime_f = save_ex_img(adv_lime_ex_img, "adv_lime.png")

    return json.dumps({"nor_result": {"nor_classname": nor_class_name,
                                      "heatmap_url": nor_h_f,
                                      "lrp_url": nor_l_f,
                                      "lime_url": nor_lime_f},
                       "adv_result": {"adv_classname": adv_class_name,
                                      "heatmap_url": adv_h_f,
                                      "lrp_url": adv_l_f,
                                      "lime_url": adv_lime_f
                                      }})


@api.route("/download", methods=["GET"])
def download():
    url = plot_overview()
    return api.send_static_file(url)


@api.route("/adverse", methods=["POST"])
def generage_adversary():
    nor_img_f = request.files.get("nor_img")
    nor_img = nor_img_f.stream
    model = request.form.get("model")
    method = request.form.get("method")
    ep = request.form.get("epsilon")
    adv_img = fool(nor_img, model, method, ep)
    adv_img_f = save_adv_img(adv_img, "adv_img.png")
    return json.dumps({"adv": adv_img_f})

@api.route("/show", methods=["GET"])
def main_show():
    ex_method = request.form.get("ex_method")
    path = "web/static/images/main_images"
    url = os.path.join(path,ex_method)

    return url
