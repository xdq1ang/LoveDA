from data.loveda import LoveDALoader
from utils.tools import *
from skimage.io import imsave
import os


def predict_test(model, cfg, ckpt_path=None, save_dir='./submit_test'):
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

    mask_save_dir = os.path.join(save_dir, "mask")
    mask_colorful_save_dir = os.path.join(save_dir, "mask_colorful")
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(mask_colorful_save_dir, exist_ok=True)
    viz_op = VisSeg(palette, mask_colorful_save_dir)

    seed_torch(2333)
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict,  strict=True)

    # count_model_parameters(model)
    model.eval()
    print(cfg.EVAL_DATA_CONFIG)
    eval_dataloader = LoveDALoader(cfg.TEST_DATA_CONFIG)

    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)
            cls = cls.argmax(dim=1).cpu().numpy()
            for fname, pred in zip(ret_gt['fname'], cls):
                imsave(os.path.join(mask_save_dir, fname), pred.astype(np.uint8))
                vis_mask = viz_op.saveVis(pred.astype(np.uint8),  fname)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    ckpt_path = './log/iast/2urban/fcD_denseppm_index_1_2/URBAN9500.pth'
    from module.Encoder import Deeplabv2
    cfg = import_config('st.iast.2urban')
    # model = Deeplabv2(dict(
    #     backbone=dict(
    #         resnet_type='resnet50',
    #         output_stride=16,
    #         pretrained=True,
    #     ),
    #     multi_layer=False,
    #     cascade=False,
    #     use_ppm=True,
    #     ppm=dict(
    #         num_classes=cfg.NUM_CLASSES,
    #         use_aux=False,
    #     ),
    #     inchannels=2048,
    #     num_classes=cfg.NUM_CLASSES
    # )).cuda()

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
    predict_test(model, cfg, ckpt_path)