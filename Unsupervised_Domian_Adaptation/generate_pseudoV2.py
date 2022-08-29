from tqdm import tqdm
import numpy as np
import os
import torch
from collections import Counter
from skimage.io import imsave
from utils.VisSeg import VisSeg
from collections import OrderedDict
from module.Encoder import Deeplabv2
from data.loveda import LoveDALoader
from configs.ToURBAN import EVAL_DATA_CONFIG
import wandb

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)
palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()

PSEIDO_DICT = dict(
    pl_alpha=0.2,
    pl_gamma=8.0,
    pl_beta=0.9
)


def generate_pseudoV2(model, target_loader, save_dir, step, n_class=7, pseudo_dict=dict(), logger=None):
    frames = []
    if logger != None:
        logger.info('Start generate pseudo labels: %s' % save_dir)
    # viz_op = er.viz.VisualizeSegmm(os.path.join(save_dir, 'vis'), palette)
    # 标签可视化类
    viz_op = VisSeg(palette, os.path.join(save_dir, 'vis'))
    os.makedirs(os.path.join(save_dir, 'pred'), exist_ok=True)
    # 模型验证模式，不会更新参数
    model.eval()
    # 概率阈值为0.9
    cls_thresh = np.ones(n_class) * 0.9
    for image, labels in tqdm(target_loader):
        out = model(image.cuda())
        # 如果out是tuple则取第0个元素，不是则直接取out
        logits = out[0] if isinstance(out, tuple) else out
        max_items = logits.max(dim=1)
        # 预测结果
        label_pred = max_items[1].data.cpu().numpy()
        logits_pred = max_items[0].data.cpu().numpy()

        # 把每个像素的预测概率保存到字典中
        # { 0: [0.9],
        #  1: [0.9, 0.1],
        #  2: [0.9, 0.2],
        #  3: [0.9, 0.3],
        #  4: [0.9, 0.4],
        #  5: [0.9, 0.5, 0.6],
        #  6: [0.9]}
        logits_cls_dict = {c: [cls_thresh[c]] for c in range(n_class)}
        # 把在dim=1维度上的类最大概率保存在字典中
        for cls in range(n_class):
            logits_cls_dict[cls].extend(logits_pred[label_pred == cls].astype(np.float16))
        # instance adaptive selector 实例自适应选择
        tmp_cls_thresh = ias_thresh(logits_cls_dict, n_class, pseudo_dict['pl_alpha'], w=cls_thresh,
                                    gamma=pseudo_dict['pl_gamma'])
        beta = pseudo_dict['pl_beta']
        cls_thresh = beta * cls_thresh + (1 - beta) * tmp_cls_thresh
        cls_thresh[cls_thresh >= 1] = 0.999

        np_logits = logits.data.cpu().numpy()
        for _i, fname in enumerate(labels['fname']):
            # save pseudo label
            logit = np_logits[_i].transpose(1, 2, 0)
            # 像素值最大的通道
            label = np.argmax(logit, axis=2)
            # 每个像素点通道维度上的最大值
            logit_amax = np.amax(logit, axis=2)
            # 如果某个像素点的所有通道的概率值都小于label_cls_thresh中特定阈值，则忽略该像素点(标签设置为0)
            label_cls_thresh = np.apply_along_axis(lambda x: [cls_thresh[e] for e in x], 1, label)
            ignore_index = logit_amax < label_cls_thresh

            # 求第2大概率
            logit_2max = np.sort(logit, axis=2)[:, :, -2]
            # 第1大概率 - 第2大概率
            logit12_sub = logit_amax - logit_2max
            # 第2大索引
            # logit_2max_index2 = np.argsort(logit, axis=2)[:,:,-2]

            # 概率差<0.2则忽略该像素
            ignore_index2 = logit12_sub < 0.1

            label += 1
            label[ignore_index] = 0
            label[ignore_index2] = 0
            # 标签保存(0---7)
            imsave(os.path.join(save_dir, 'pred', fname), label.astype(np.uint8))
            # 可视化标签保存(0---6)
            vis_mask = label.copy()
            vis_mask[vis_mask == 0] = 1 # 为了方便观察，ignore设置为背景
            vis_mask -= 1
            vis_mask = viz_op.saveVis(vis_mask, fname)
            frames.append(wandb.Image(vis_mask, caption=fname))
    wandb.log({"pseudo_label": frames}, step=step)
    return os.path.join(save_dir, 'pred')


def ias_thresh(conf_dict, n_class, alpha, w=None, gamma=1.0):
    # 如果权重不存在，则自动生成为[1,1,1,1,1,1,1]
    # 此项目中 
    # conf_dict形如 = {  0: [0.9],
    #                1: [0.9, 0.1],
    #                2: [0.9, 0.2],
    #                3: [0.9, 0.3],
    #                4: [0.9, 0.4],
    #                5: [0.9, 0.5, 0.6],
    #                6: [0.9]  }
    # w = cls_thresh = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9] 
    # n_class = 7
    # alpha = 0.2
    # gamma = 8.0
    if w is None:
        w = np.ones(n_class)
    # threshold
    cls_thresh = np.ones(n_class, dtype=np.float32)
    for idx_cls in np.arange(0, n_class):
        if conf_dict[idx_cls] != None:
            arr = np.array(conf_dict[idx_cls])
            # 计算分位数
            tmp = 100 * (1 - alpha * w[idx_cls] ** gamma)
            # 含义为计算数组中 tmp%的数的值
            cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
    return cls_thresh


def getModel(model_name):
    if (model_name == "IAST_DensePPM"):
        model = Deeplabv2(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=False,
            cascade=False,
            use_ppm='denseppm',
            ppm=dict(
                in_channels=2048,
                num_classes=7,
                reduction_dim=64,
                pool_sizes=[2, 3, 4, 5]
            ),
            inchannels=2048,
            num_classes=7
        )).cuda()
    elif model_name == "IAST_PPM":
        model = Deeplabv2(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=False,
            cascade=False,
            use_ppm='ppm',
            ppm=dict(
                num_classes=7,
                use_aux=False,
            ),
            inchannels=2048,
            num_classes=7
        )).cuda()
    return model


if __name__ == '__main__':
    pseudo_dir = r"utils\pseudoTempFile"
    ckpt_path = r"log\iast_training_step_20000_denseppm\2urban\URBAN20000.pth"
    evalloader = LoveDALoader(EVAL_DATA_CONFIG)

    model = getModel("IAST_DensePPM")
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict, strict=True)

    pseudo_pred_dir = generate_pseudoV2(model,
                                        evalloader,
                                        pseudo_dir,
                                        pseudo_dict=PSEIDO_DICT)
