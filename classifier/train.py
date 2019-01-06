import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from sklearn.metrics import confusion_matrix

import time
from termcolor import colored as clr

from dataset import NTUSkeletonDataset
import dataset.transforms as skeleton_transforms

sep = clr(" |\t", "white")


def color_form(text, param_format, color, param):
    str_ = text.format(
        clr(param_format.format(param), color)
    )
    return str_


def report_net_param(model):
    param_avg, grad_avg = .0, .0
    param_max, grad_max = None, None
    param_groups_count = .0
    for p in model.parameters():
        if p.grad is not None:
            param_avg += p.data.abs().mean()
            grad_avg += p.grad.data.abs().mean()
            p_max = p.data.abs().max()
            g_max = p.grad.data.abs().max()
            param_max = max(p_max, param_max) if param_max else p_max
            grad_max = max(g_max, grad_max) if grad_max else g_max
            param_groups_count += 1

    param_avg = param_avg / param_groups_count
    grad_avg = grad_avg / param_groups_count

    net_info = clr("[Net param] ", "red") + color_form("Max param: {:s}",
                                                       "{:4.8f}", "green",
                                                       param_max)
    net_info += sep + color_form("Avg param: {:s}", "{:4.8f}", "green",
                                 param_avg)
    net_info += sep + color_form("Max grad: {:s}", "{:4.8f}", "green",
                                 grad_max)
    net_info += sep + color_form("Avg grad: {:s}", "{:4.8f}", "green",
                                 grad_avg)

    net_info_str = "[Net param] Max param: {:4.8f}\tAvg param: {:4.8f}" \
                   "\tMax grad: {:4.8f}\tAvg grad: {:4.8f}" \
        .format(param_max, param_avg, grad_max, grad_avg)

    return net_info, net_info_str, (param_max, param_avg, grad_max, grad_avg)


def train(args, model, optimizer, loss, log_file=None, test_cnf=None, val_cnf=None,
          models_folder=None, tb_writer=None):
    log = log_file
    start = time.time()
    train_set = NTUSkeletonDataset.NTUSkeletonDataset(
        args['train_path'],
        cache_dir=args['cache_path'],
        selected_actions=args['selected_actions'],
        selected_joints=args['selected_joints'],
        transform=transforms.Compose([
            skeleton_transforms.MoveOriginToJoint(),
            skeleton_transforms.GaussianFilter(),
            skeleton_transforms.ResizeSkeletonSegments(),
            skeleton_transforms.UniformSampleOrPad(args['maximum_sample_size']),
            skeleton_transforms.ToTensor(),
            skeleton_transforms.MovingPoseDescriptor(args['maximum_sample_size'])
        ]),
        use_cache=args['use_cache'],
        use_validation=args['use_validation'],
        validation_fraction=args['validation_fraction'],
        preprocessing_threads=args['preprocessing_threads'])
    end = time.time()
    num_train_samples = len(train_set)
    train_set.set_use_mode(NTUSkeletonDataset.DatasetMode.VALIDATION)
    num_val_samples = len(train_set)
    train_set.set_use_mode(NTUSkeletonDataset.DatasetMode.TRAIN)
    print(end - start, "Loaded {} train samples and {} validation samples".
          format(num_train_samples, num_val_samples))

    start = time.time()
    test_set = NTUSkeletonDataset.NTUSkeletonDataset(
        args['test_path'],
        cache_dir=args['cache_path'],
        selected_actions=args['selected_actions'],
        selected_joints=args['selected_joints'],
        transform=transforms.Compose([
            skeleton_transforms.MoveOriginToJoint(),
            skeleton_transforms.GaussianFilter(),
            skeleton_transforms.ResizeSkeletonSegments(),
            skeleton_transforms.UniformSampleOrPad(args['maximum_sample_size']),
            skeleton_transforms.ToTensor(),
            skeleton_transforms.MovingPoseDescriptor(args['maximum_sample_size'])
        ]),
        use_cache=args['use_cache'],
        use_validation=False,
        preprocessing_threads=args['preprocessing_threads'])
    end = time.time()
    print(end - start, "Loaded {} test samples".format(len(test_set)))
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              shuffle=True)

    min_train_loss = np.inf
    min_train_epoch = -1

    min_validation_loss = np.inf
    min_validation_epoch = -1
    max_validation_acc = 0
    max_validation_acc_epoch = -1

    min_test_loss = np.inf
    min_test_epoch = -1
    max_test_acc = 0
    max_test_acc_epoch = -1

    for epoch in range(1, args['epochs'] + 1):
        __log('\n################\n### EPOCH {}\n################\n'
              .format(epoch), color='cyan', log=log)

        train_loss, (mean_max_param, mean_avg_param, mean_max_grad,
                     mean_avg_grad) = train_epoch(epoch, args, model,
                                                  train_loader, optimizer, loss, log)

        if train_loss < min_train_loss:
            color = "green"
            min_train_loss = train_loss
            min_train_epoch = epoch
        else:
            color = "red"

        __log('[TRAIN] Mean loss: {}\t'
              'Best_train_loss: {}\t at epoch: {}'
              .format(train_loss, min_train_loss, min_train_epoch),
              color=color, log=log)
        if epoch % 5 != 0:
            continue

        # Perform validation
        train_set.set_use_mode(NTUSkeletonDataset.DatasetMode.VALIDATION)
        validation_loss, validation_acc = validate(epoch, model, train_set,
                                                   args, val_cnf, loss, log)
        train_set.set_use_mode(NTUSkeletonDataset.DatasetMode.TRAIN)
        test_color = None
        test_loss = 0.
        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            min_validation_epoch = epoch
        if validation_acc > max_validation_acc:
            color = "green"
            max_validation_acc = validation_acc
            max_validation_acc_epoch = epoch

            # Perform test on best models
            test_loss, test_acc = test(epoch, model, test_set, args, test_cnf, loss)
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                min_test_epoch = epoch
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                max_test_acc_epoch = epoch
                test_color = "blue"
                save_best_model(model, models_folder)
            else:
                test_color = "red"
        else:
            color = "red"

        __log('[VALIDATION] Mean loss\t: {}\t Best_validation_loss: {}\t'
              'at epoch: {}'
              .format(validation_loss, min_validation_loss,
                      min_validation_epoch),
              log=log)
        __log('[VALIDATION] Accuracy\t: {}\t Best_validation_acc: {}\t '
              'at epoch: {}'
              .format(validation_acc, max_validation_acc,
                      max_validation_acc_epoch),
              color=color,
              log=log)

        if test_color:
            __log('[TEST] Mean loss\t: {}\t Best_test_loss: {}\t'
                  'at epoch: {}'
                  .format(test_loss, min_test_loss,
                          min_test_epoch),
                  log=log)
            __log('[TEST] Accuracy\t: {}\t Best_test_acc: {}\t '
                  'at epoch: {}'
                  .format(test_acc, max_test_acc, max_test_acc_epoch),
                  color=test_color,
                  log=log)

        if test_color:
            tb_writer.add_scalars('Loss',
                                  {'Train': train_loss,
                                   'Validation': validation_loss,
                                   'Test': test_acc},
                                  epoch
                                  )
            tb_writer.add_scalar('Test-Accuracy', test_acc, epoch)
        else:
            tb_writer.add_scalars('Loss',
                                  {'Train': train_loss,
                                   'Validation': validation_loss},
                                  epoch
                                  )
        tb_writer.add_scalar('Validation-Accuracy', validation_acc, epoch)
        tb_writer.add_scalars('Params',
                              {'Max': mean_max_param,
                               'Avg': mean_avg_param
                               },
                              epoch)
        tb_writer.add_scalars('Grads',
                              {'Max': mean_max_grad,
                               'Avg': mean_avg_grad
                               },
                              epoch)

        log.flush()


def train_epoch(epoch, args, model, train_loader, optimizer, loss, log=None):
    epoch_loss = 0
    # Switch to train mode
    model.train()

    mean_max_param = 0.
    mean_avg_param = 0.
    mean_max_grad = 0.
    mean_avg_grad = 0.

    for i_batch, batch in enumerate(train_loader):
        if args['use_sampling']:
            batch = skeleton_transforms.UniformSample(args['sample_size'])(batch)

        # Sort samples from batch by length (descending)
        sorted_lengths, indices = torch.sort(batch['length'], -1,
                                             descending=True)
        batch['length'] = sorted_lengths
        batch['frames'] = batch['frames'][indices].squeeze()
        batch['tag'] = batch['tag'][indices]
        batch['filename'] = [batch['filename'][i] for i in indices]
        target = Variable(batch['tag'].type(torch.LongTensor).cuda(),
                          requires_grad=False)

        # Compute prediction and gradient
        optimizer.zero_grad()
        output = model(Variable(batch['frames'].type(torch.FloatTensor).cuda(),
                                requires_grad=False),
                       batch['length'].numpy())
        err = loss(output, target)
        epoch_loss += err.data.item()
        err.backward()

        # Clamp gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-0.3, 0.3)

        #  Do optimization step
        optimizer.step()

        if i_batch % args['log_interval'] == 0:
            # Measure model parameters and record loss
            print_info, print_info_str, (param_max, param_avg, grad_max,
                                         grad_avg) = report_net_param(model)
            mean_max_param += param_max
            mean_avg_param += param_avg
            mean_max_grad += grad_max
            mean_avg_grad += grad_avg
            if log:
                log.write(print_info_str + '\n')
            print(print_info)
            __log('[TRAIN] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i_batch * args['batch_size'], len(train_loader.dataset),
                100. * i_batch / len(train_loader), err.data.item()),
                log=log)

    # Compute mean loss
    batch_count = len(train_loader)
    mean_loss = epoch_loss / batch_count
    return mean_loss, (mean_max_param / batch_count,
                       mean_avg_param / batch_count,
                       mean_max_grad / batch_count,
                       mean_avg_grad / batch_count)


def validate(epoch, model, validation_set, args, val_cnf, loss, log=None):
    validation_loss = 0
    correct = 0
    i_sample = 0
    num_samples = len(validation_set)
    num_classes = 0
    targets = []
    predictions = []

    # Switch to evaluate mode
    model.eval()

    while i_sample < num_samples:
        # If the sample is a sequence coming from a multi-user action
        if i_sample < num_samples - 1 and validation_set[i_sample]['filename'] \
                == validation_set[i_sample + 1]['filename']:
            sample1 = validation_set[i_sample]
            sample2 = validation_set[i_sample + 1]

            if args['use_sampling']:
                sample1 = skeleton_transforms.UniformSample(args['sample_size'],
                                                            use_batch=False)(
                    sample1)
                sample2 = skeleton_transforms.UniformSample(args['sample_size'],
                                                            use_batch=False)(
                    sample2)

            target = Variable(torch.LongTensor([sample1['tag']]).cuda(),
                              requires_grad=False)
            sample_size = sample1['frames'].size()

            # Pass both sequences in the sample through the model
            output1 = model(Variable(
                sample1['frames'].type(torch.FloatTensor).view(1,
                                                               sample_size[0],
                                                               sample_size[
                                                                   2])).cuda(),
                            np.array([sample1['length']]))
            output2 = model(Variable(
                sample2['frames'].type(torch.FloatTensor).view(1,
                                                               sample_size[0],
                                                               sample_size[
                                                                   2])).cuda(),
                            np.array([sample2['length']]))

            # Average the results from the two sequences (one per each user)
            output = (output1 + output2) / 2
            i_sample += 2
        else:
            sample = validation_set[i_sample]

            if args['use_sampling']:
                sample = skeleton_transforms.UniformSample(args['sample_size'],
                                                           use_batch=False)(
                    sample)
            sample_size = sample['frames'].size()
            target = Variable(torch.LongTensor([sample['tag']]).cuda(),
                              requires_grad=False)
            output = model(Variable(
                sample['frames'].type(torch.FloatTensor).view(1, sample_size[0],
                                                              sample_size[
                                                                  2])).cuda(),
                           np.array([sample['length']]))
            i_sample += 1

        num_classes += 1
        err = loss(output, target)
        validation_loss += err.data.item()

        # Get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        predictions.append(pred[0])
        targets.append(target.data[0])

    # Compute and save the validation confusion matrix
    cnf = confusion_matrix(targets, predictions)
    cnf = cnf.astype('float') / cnf.sum(axis=1)[:, np.newaxis]
    np.savetxt(val_cnf + '_' + str(epoch) + '.csv', cnf,
               delimiter=',', fmt='%.2f')

    # Compute and return mean loss and accuracy
    validation_loss /= num_classes
    correct = correct.numpy()
    acc = (100.0 * correct) / num_classes
    return validation_loss, acc


def test(epoch, model, test_set, args, test_cnf, loss, log=None):
    test_loss = 0.
    correct = 0
    i_sample = 0
    num_samples = len(test_set)
    num_classes = 0
    targets = []
    predictions = []

    # Switch to evaluate mode
    model.eval()

    while i_sample < num_samples:
        # If the sample is a sequence coming from a multi-user action
        if i_sample < num_samples - 1 and test_set[i_sample]['filename'] \
                == test_set[i_sample + 1]['filename']:
            sample1 = test_set[i_sample]
            sample2 = test_set[i_sample + 1]

            if args['use_sampling']:
                sample1 = skeleton_transforms.UniformSample(args['sample_size'],
                                                            use_batch=False)(
                    sample1)
                sample2 = skeleton_transforms.UniformSample(args['sample_size'],
                                                            use_batch=False)(
                    sample2)

            target = Variable(torch.LongTensor([sample1['tag']]).cuda(),
                              requires_grad=False)
            sample_size = sample1['frames'].size()

            # Pass both sequences in the sample through the model
            output1 = model(Variable(
                sample1['frames'].type(torch.FloatTensor).view(1,
                                                               sample_size[0],
                                                               sample_size[
                                                                   2])).cuda(),
                            np.array([sample1['length']]))
            output2 = model(Variable(
                sample2['frames'].type(torch.FloatTensor).view(1,
                                                               sample_size[0],
                                                               sample_size[
                                                                   2])).cuda(),
                            np.array([sample2['length']]))

            # Average the results from the two sequences (one per each user)
            output = (output1 + output2) / 2
            i_sample += 2
        else:
            sample = test_set[i_sample]

            if args['use_sampling']:
                sample = skeleton_transforms.UniformSample(args['sample_size'],
                                                           use_batch=False)(
                    sample)
            sample_size = sample['frames'].size()
            target = Variable(torch.LongTensor([sample['tag']]).cuda(),
                              requires_grad=False)
            output = model(Variable(
                sample['frames'].type(torch.FloatTensor).view(1, sample_size[0],
                                                              sample_size[
                                                                  2])).cuda(),
                           np.array([sample['length']]))
            i_sample += 1

        num_classes += 1
        err = loss(output, target)
        test_loss += err.data.item()
        # Get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        predictions.append(pred[0])
        targets.append(target.data[0])

    # Compute and save the validation confusion matrix
    cnf = confusion_matrix(targets, predictions)
    cnf = cnf.astype('float') / cnf.sum(axis=1)[:, np.newaxis]
    np.savetxt(test_cnf + '_' + str(epoch) + '.csv', cnf,
               delimiter=',', fmt='%.2f')

    # Compute and return mean loss and accuracy
    test_loss /= num_classes
    correct = correct.numpy()
    test_acc = (100.0 * correct) / num_classes
    return test_loss, test_acc


def checkpoint(model, epoch, models_folder, log=None):
    model_out_path = models_folder + '/model_chk_epoch_{}.pth'.format(epoch)
    torch.save(model, model_out_path)
    __log('[CHECKPOINT] Checkpoint saved to {}'.format(model_out_path), log=log)


def save_best_model(model, models_folder, log=None):
    model_out_path = models_folder + '/best_model.pth'
    torch.save(model, model_out_path)
    __log('[CHECKPOINT] Best model saved to {}'.format(model_out_path), log=log)


def __log(line, color=None, log=None):
    if log:
        log.write(line + '\n')
    print(clr(line, color=color))
