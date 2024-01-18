import argparse
import yaml
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5, 6" 

import torch
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter
from utils.utils import get_writer_path

from utils.loader import data_loader, get_module,  get_save_path
from utils.logging import *


def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches' % \
                 (tag, len(train_loader) * config['model']['batch_size'], len(train_loader)))


def train_joint(config, output_dir, args):
    assert 'train_iter' in config

    torch.set_default_tensor_type(torch.FloatTensor)
    task = config['data']['dataset']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    writer = SummaryWriter(get_writer_path(task=args.command, exper_name=args.exper_name, date=False))
    save_path = get_save_path(output_dir)

    data = data_loader(config, dataset=task, warp_input=True)
    train_loader, val_loader = data['train_loader'], data['val_loader']

    datasize(train_loader, config, tag='train')
    datasize(val_loader, config, tag='val')

    # init the training agent using config file
    # from train_model_frontend import Train_model_frontend

    train_model_frontend = get_module('trian_process', config['front_end_model'])

    train_agent = train_model_frontend(config, save_path=save_path, device=device)

    # writer from tensorboard
    train_agent.writer = writer

    # feed the data into the agent
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader

    # load model initiates the model and load the pretrained model (if any)
    train_agent.loadModel()
    train_agent.dataParallel()

    try:
        # train function takes care of training and evaluation
        train_agent.train()
    except KeyboardInterrupt:
        print("press ctrl + c, save model!")
        train_agent.saveModel()
        pass


if __name__ == '__main__':
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # add parser
    p_train = argparse.ArgumentParser()

    # Training command
    p_train.add_argument("--command", type=str, default='train_joint')
    p_train.add_argument('--config', type=str, default='configs/superpoint_simcol_train_heatmap.yaml')

    
    p_train.add_argument('--exper_name', type=str, default='superpoint_simcol/')
    p_train.add_argument('--eval', action='store_true', default=True)
    p_train.add_argument('--debug', action='store_true', default=True, help='turn on debuging mode')
    p_train.add_argument("--output_path", type=str, default="/data/hyliu/simcol_output/")
    p_train.set_defaults(func=train_joint)

    args = p_train.parse_args()

    if args.debug:
        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = args.output_path + args.exper_name
    os.makedirs(output_dir, exist_ok=True)

    logging.info('Running command {}'.format(args.command.upper()))
    args.func(config, output_dir, args)

