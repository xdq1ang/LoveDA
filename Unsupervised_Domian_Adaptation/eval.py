from data.loveda import LoveDALoader
import logging
logger = logging.getLogger(__name__)
from utils.tools import *
from ever.util.param_util import count_model_parameters
from module.viz import VisualizeSegmm
import wandb
from PIL import ImagePalette
from torch import nn



def evaluate(model, model_D, cfg, step, is_training=False, ckpt_path=None, logger=None):
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
    with torch.no_grad():
        frames = []
        if cfg.SNAPSHOT_DIR is not None:
            predict_vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
            domain_vis_cls_dir = os.path.join(cfg.SNAPSHOT_DIR, 'dom_cls-{}'.format(os.path.basename(ckpt_path)))
            domain_vis_dis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'dom_dis-{}'.format(os.path.basename(ckpt_path)))
            palette_predict = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
            palette_domain_cls = np.asarray(list(DOMAIN_CLS_COLOR_MAP.values())).reshape((-1,)).tolist()
            palette_domain_dis = ImagePalette.ImagePalette(mode='RGB', palette=None, size=0)
            viz_predict = VisualizeSegmm(predict_vis_dir, palette_predict)
            if (model_D != None):
                viz_domain_cls = VisualizeSegmm(domain_vis_cls_dir, palette_domain_cls)
                viz_domain_dis = VisualizeSegmm(domain_vis_dis_dir, palette_domain_dis)
        if not is_training:
            model_state_dict = torch.load(ckpt_path)
            model.load_state_dict(model_state_dict,  strict=True)
            logger.info('[Load params] from {}'.format(ckpt_path))
            count_model_parameters(model, logger)
        model.eval()
        if (model_D != None):
            model_D.eval()
        print(cfg.EVAL_DATA_CONFIG)
        eval_dataloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
        metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
        mIoU=[]
        with torch.no_grad():
            for ret, ret_gt in tqdm(eval_dataloader):
                ret = ret.to(torch.device('cuda'))
                cls = model(ret)
                if (model_D != None):
                    t_D_logits = model_D(cls.softmax(dim=1).detach())
                    t_D_logits = nn.Sigmoid()(t_D_logits)
                    domain_cls = torch.zeros_like(t_D_logits)
                    domain_cls[t_D_logits >= 0.5] = 1
                    domain_cls = domain_cls.cpu().detach().numpy()
                    domain_dis = t_D_logits.cpu().detach().numpy()
                cls = cls.argmax(dim=1).cpu().numpy()

                cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
                mask = cls_gt >= 0

                y_true = cls_gt[mask].ravel()
                y_pred = cls[mask].ravel()
                metric_op.forward(y_true, y_pred)
                # 计算miou
                confusion_matrix = metric_op._total.toarray()
                iou_per_class = metric_op.compute_iou_per_class(confusion_matrix)
                miou = iou_per_class.mean()
                mIoU.append(miou)
                save_domain_dis = True
                save_domain_cls = True
                if cfg.SNAPSHOT_DIR is not None:
                    if (model_D != None):
                        for fname, pred, d_dis, d_cls in zip(ret_gt['fname'], cls, domain_dis, domain_cls):
                            viz_img = viz_predict.setpalette(pred, fname.replace('tif', 'png'))
                            if save_domain_dis:
                                viz_domian_dis_img = viz_domain_dis.saveheatmap(d_dis[0], fname.replace('tif', 'png'))
                            if save_domain_cls:
                                viz_domian_cls_img = viz_domain_cls.setpalette(d_cls[0], fname.replace('tif', 'png'))
                            frames.append(wandb.Image(viz_img, caption=fname))
                    else:
                        for fname, pred in zip(ret_gt['fname'], cls):
                            viz_img = viz_predict.setpalette(pred, fname.replace('tif', 'png'))
                            frames.append(wandb.Image(viz_img, caption=fname))
            if is_training:
                wandb.log({"prediction": frames}, step=step)

        metric_op.summary_all()
        torch.cuda.empty_cache()
        return np.nanmean(np.array(mIoU))



if __name__ == '__main__':
    seed_torch(2333)
    ckpt_path = './log/iast_training_step_20000/2urban/URBAN20000.pth'
    from module.Encoder import Deeplabv2
    cfg = import_config('st.iast.2urban')
    logger = get_console_file_logger(name='Baseline', logdir=cfg.SNAPSHOT_DIR)
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
    evaluate(model, cfg, False, ckpt_path, logger)