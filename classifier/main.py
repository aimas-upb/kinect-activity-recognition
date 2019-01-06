import os
import time
import random
import yaml
import torch
import torch.optim as optim
import numpy
from tensorboardX import SummaryWriter
import zipfile
from train import train


def import_class(name):
    """Returns a class from a string including module and class."""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':

    # Load the config file (located at the same path with this file)
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Set all the seeds
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    numpy.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    numpy.random.seed(cfg['seed'])

    # Set data properties
    joint_size = cfg['joint_size']
    if cfg['selected_joints']:
        in_size = joint_size * len(cfg['selected_joints'])
        no_of_joints = len(cfg['selected_joints'])
    else:
        in_size = joint_size * 25
        no_of_joints = 25
    hidden_dim = cfg['hidden_dim']
    num_classes = 60
    if cfg['selected_actions']:
        num_classes = len(cfg['selected_actions'])

    # Set up model
    Model = import_class(cfg['model'])
    model = Model(**cfg['model_args']).cuda()
    model.share_memory()

    # Set optimizer properties
    optimizer_args = cfg['optimizer_args']
    learning_rate = optimizer_args['learning_rate']

    # Set up optimizer
    if cfg['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif cfg['optimizer'] == 'SGD':
        if 'momentum' in optimizer_args:
            momentum = optimizer_args['momentum']
        else:
            momentum = 0.9
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Set up loss function
    Loss = import_class(cfg['loss'])
    loss = Loss()

    # Set up experiment folders
    exp_time = time.strftime('%Y_%m_%d_%H:%M:%S')
    exp_folder = '../skeleton-experiments/exp_' + exp_time
    fig_folder = exp_folder + '/figures'
    models_folder = exp_folder + '/models'
    cnf_folder = exp_folder + '/cnf'
    arch_file = exp_folder + '/architecture.txt'
    conf_file = exp_folder + '/config.yml'
    test_cnf = exp_folder + '/test_confmat'
    val_cnf = exp_folder + '/val_confmat'
    os.mkdir(exp_folder)
    os.mkdir(models_folder)

    # Write architecture and config files
    f = open(arch_file, 'w')
    f.write(str(model) + '\n')
    f.close()
    with open(conf_file, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    zf = zipfile.ZipFile(exp_folder + '/source_code.zip', mode='w')
    for dirname, _, files in os.walk('.'):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

    # Set up log file
    log_file = open(exp_folder + '/output.log', 'w')

    # Set up Tensorboard
    tb_folder = '../skeleton-tb-logs/exp_' + exp_time
    os.mkdir(tb_folder)
    tb_writer = SummaryWriter(log_dir=tb_folder)

    # Starting the training process
    train(cfg, model, optimizer, loss, log_file, test_cnf, val_cnf,
          models_folder, tb_writer)
