import sys, os, shutil
import torch
sys.path.append('.')

import test_utils

from datasets import custom_data_loader
from options  import run_model_opts
from models import custom_model
from utils  import logger, recorders

args = run_model_opts.RunModelOpts().parse()
log  = logger.Logger(args) # Namespace(benchmark='DiLiGenT_main', bm_dir='data/datasets/DiLiGenT/pmsData', cuda=True, epochs=30, fuse_type='max', in_img_num=96, in_light=True, item='calib', model='PS_FCN_run', normalize=False, resume=None, retrain='data/models/PS-FCN_B_S_32.pth.tar', run_model=True, save_root='data/Training/', seed=0, start_epoch=1, test_batch=1, test_disp=1, test_intv=1, test_save=1, time_sync=False, train_img_num=32, use_BN=False, workers=8)

def main(args):
    test_loader = custom_data_loader.benchmarkLoader(args)
    model    = custom_model.buildModel(args)
    recorder = recorders.Records(args.log_dir)
    test_utils.test(args, 'test', test_loader, model, log, 1, recorder)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
