import argparse
import torch
import torch.optim as optim
import os.path as osp
from module.Encoder import Deeplabv2
from data.loveda import LoveDALoader
from ever.core.iterator import Iterator
from utils.tools import *
from torch.nn.utils import clip_grad
import torch.nn.functional as F
from tqdm import tqdm
from eval import evaluate
import torch.nn as nn
import wandb
import torch.backends.cudnn as cudnn
parser = argparse.ArgumentParser(description='Run Baseline methods.')

parser.add_argument('--config_path',  type=str,
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)



def main():
    # 初始化wandb
    wandbLogger = wandb.init(
        project="UDA",
        notes="DeepLabV2_PPM",
        tags=["领域自适应", "语义分割"],
        resume="allow",
    )
    """Create the model and start the training."""
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='Deeplabv2', logdir=cfg.SNAPSHOT_DIR)
    # Create Network
    
    # model = Deeplabv2(dict(
    #   backbone=dict(
    #             resnet_type='resnet50',
    #             output_stride=16,
    #             pretrained=True,
    #         ),
    #         multi_layer=False,
    #         cascade=False,
    #         use_ppm="ppm",
    #         ppm=dict(
    #             num_classes=7,
    #             use_aux=False,
    #             norm_layer=nn.BatchNorm2d,
    #         ),
    #         inchannels=2048,
    #         num_classes=7
    # ))
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
    ))
    model.train()
    model.cuda()
    #cudnn.enabled = True
    #cudnn.benchmark = True
    logger.info('exp = %s'% cfg.SNAPSHOT_DIR)
    count_model_parameters(model, logger)
    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)
    trainloader_iter = Iterator(trainloader)
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    optimizer.zero_grad()

    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter, cfg)
        # Train with Source
        batch = trainloader_iter.next()
        images_s, labels_s = batch[0]
        pred_source = model(images_s.cuda())
        pred_source = pred_source[0] if isinstance(pred_source, tuple) else pred_source
        #Segmentation Loss
        loss = loss_calc(pred_source, labels_s['cls'].cuda())
        
        loss.backward()
        
        optimizer.step()
        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            text = 'iter = %d, loss_seg = %.3f, lr = %.3f'% (
                i_iter, loss, lr)
            logger.info(text)
            wandbLogger.log({'src_seg_loss': loss, "seg_model_lr": lr})
        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            miou = evaluate(model, cfg, True, ckpt_path, logger)
            wandbLogger.log({'src_seg_loss': loss, 'tar_mIoU': miou})
            break
        if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            miou = evaluate(model, cfg, True, ckpt_path, logger)
            wandbLogger.log({'src_seg_loss': loss, 'tar_mIoU': miou})
            model.train()



if __name__ == '__main__':
    seed_torch(2333)
    main()
