import yaml
import time
import os
import copy
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_warmup as warmup
from utils.general import colorstr, save_ckpt_, plot_and_log_result
from config.vgg_configs import feature_index
from models.vgg import get_model
from models.utils import download_ckpt
from models.Converter import Converter
from torchsummary import summary


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls")


def train_model(opt, dataloaders, wandb):
    since = time.perf_counter()

    LOGGER.info(f"\n{colorstr('Hyperparameter:')} {opt}")

    num_epochs, device = opt.epochs, opt.device

    LOGGER.info(f"\n{colorstr('Device:')} {device}")

    if opt.log_result:
        DIR_SAVE = "./runs/train/{}/run_seed_{}".format(opt.name, opt.seed)
        os.makedirs(DIR_SAVE)
        save_opt = os.path.join(DIR_SAVE, "opt.yaml")
        with open(save_opt, 'w') as f:
            yaml.dump(opt.__dict__, f, sort_keys=False)

    total_step = len(dataloaders['train']) * num_epochs
    if opt.lr_step is not None:
        milestones = []
        for i in range(len(opt.lr_step)):
            milestones.append(int(opt.lr_step[i] * total_step))

    else:  # use default
        milestones = [
            int(0.6 * total_step),
            int(0.8 * total_step),
        ]

    LOGGER.info(f"\n{colorstr('Model type:')} {opt.model_type}")
    LOGGER.info(f"\n{colorstr('Weight initialization:')} {opt.base_init}")

    model = get_model(model_type=opt.model_type,
                      base_init=opt.base_init,
                      num_classes=opt.num_classes,
                      )

    if opt.transfer_weight:
        LOGGER.info(f"\n{colorstr('Weight mapping type:')} {opt.mapping_type}")
        if opt.mapping_type == "relative":
            LOGGER.info(f"\n{colorstr('Method choose candidate:')} {opt.cand_select_method}")
            LOGGER.info(f"\n{colorstr('Number of candidate:')} {opt.num_candidate}")

        vgg16_ckpt = download_ckpt()
        converter = Converter(model = model, 
                              ckpt=vgg16_ckpt, 
                              feature_index=feature_index["vgg16"],
                              candidate_method = opt.cand_select_method,
                              mapping_type=opt.mapping_type)

        LOGGER.info(f"\n{colorstr('Method keep weight:')} {opt.keep}")
        LOGGER.info(f"\n{colorstr('Method remove weight:')} {opt.remove}")

        model = converter._load_weight(
                                        type_pad=opt.type_pad,
                                        type_pool=opt.type_pool,
                                        num_candidate=opt.num_candidate,
                                        choice_method={
                                                        "keep": opt.keep, 
                                                        "remove": opt.remove
                                                        },
                                    )

    if opt.show_summary:
        summary(model, (3, opt.image_sz, opt.image_sz))

    if opt.adam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    LOGGER.info(f"\n{colorstr('Optimizer:')} {optimizer}")

    if opt.lr_scheduler:
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones)
        LOGGER.info(
            f"\n{colorstr('LR Scheduler:')} {type(lr_scheduler).__name__}")
    else:
        lr_scheduler = None

    if opt.adam and opt.warm_up:
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
        LOGGER.info(
            f"\n{colorstr('Warm-up Scheduler:')} {type(warmup_scheduler).__name__}")
    else:
        warmup_scheduler = None

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count())))

    criterion = torch.nn.CrossEntropyLoss()
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_optim = copy.deepcopy(optimizer.state_dict())
    best_val_acc = 0.0

    model.to(device)
    for epoch in range(num_epochs):
        LOGGER.info(colorstr(f'\nEpoch {epoch}/{num_epochs-1}:'))

        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                            ('Training:', 'gpu_mem', 'loss', 'acc'))
                model.train()
            else:
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                            ('Validation:', 'gpu_mem', 'loss', 'acc'))
                model.eval()

            running_items = 0
            running_loss = 0.0
            running_corrects = 0

            _phase = tqdm(dataloaders[phase],
                          total=len(dataloaders[phase]),
                          bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                          unit='batch')

            for inputs, labels in _phase:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        history['lr'].append(lr_scheduler.optimizer.param_groups[0]
                                             ["lr"]) if lr_scheduler else history['lr'].append(opt.lr)

                        if warmup_scheduler is not None and lr_scheduler is not None:
                            with warmup_scheduler.dampening():
                                lr_scheduler.step()
                        elif warmup_scheduler is None and lr_scheduler is not None:
                            lr_scheduler.step()

                running_items += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / running_items
                epoch_acc = running_corrects / running_items

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                desc = ('%35s' + '%15.6g' * 2) % (mem, running_loss /
                                                  running_items, running_corrects / running_items)
                _phase.set_description_str(desc)

            if phase == 'train':
                if opt.wandb_log:
                    wandb.log({"train_acc": epoch_acc,
                              "train_loss": epoch_loss})
                    if lr_scheduler:
                        wandb.log(
                            {"lr": lr_scheduler.optimizer.param_groups[0]["lr"]})
                    else:
                        wandb.log({"lr": opt.lr})
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                if opt.wandb_log:
                    wandb.log({"val_acc": epoch_acc, "val_loss": epoch_loss})
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_optim = copy.deepcopy(optimizer.state_dict())
                    if opt.save_ckpt:
                        save_ckpt_(model, DIR_SAVE, "best.pt")

    time_elapsed = time.perf_counter() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s with {} epochs'.format(
        time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60, num_epochs))
    print('Best val Acc: {:4f}'.format(best_val_acc))
    if opt.save_ckpt:
        save_ckpt_(model, DIR_SAVE, "last.pt")

    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_model_optim)

    if opt.log_result:
        plot_and_log_result(DIR_SAVE, history)

    if opt.save_ckpt:
        print("Best model weight saved at {}/weights/best.pt".format(DIR_SAVE))
        print("Last model weight saved at {}/weights/last.pt".format(DIR_SAVE))

    return model, best_val_acc.item()


def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    totals = 0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            totals += inputs.size(0)
            corrects += torch.sum(preds == labels.data)

    acc = corrects / totals
    return acc.item()
