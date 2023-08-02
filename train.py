import os
import copy
import time
import logging
import argparse
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm
import wandb
import torch
from torchvision import models as torchmodel
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from utils.general import AppPath, colorstr, save_ckpt_, plot_and_log_result, seed_everything
from dataset_loader.dataset import get_train_valid_loader, get_test_loader
from models import CustomResnet

from toolkit import TLV
from toolkit.standardization import FlattenStandardization
from toolkit.matching import IndexMatching
from toolkit.transfer import VarTransfer

seed_everything(2)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls")

p_crossover  = 0.1

def train_model(model, support_model, dataloaders, optimizer, opt, wandb, lr_scheduler=None):
    since = time.perf_counter()
    num_epochs, device = opt.epochs, opt.device
    LOGGER.info(f"\n{colorstr('Hyperparameter:')} {opt}")
    LOGGER.info(f"\n{colorstr('Device:')} {device}")
    LOGGER.info(f"\n{colorstr('Optimizer:')} {optimizer}")
    if opt.log_result:
        DIR_SAVE = AppPath.RUN_TRAIN_DIR / \
            "{}/run_seed_{}".format(opt.name, opt.seed)
        os.makedirs(DIR_SAVE)
        save_opt = os.path.join(DIR_SAVE, "opt.yaml")
        with open(save_opt, 'w') as f:
            yaml.dump(opt.__dict__, f, sort_keys=False)
    if opt.lr_scheduler:
        LOGGER.info(
            f"\n{colorstr('LR Scheduler:')} {type(lr_scheduler).__name__}")
    else:
        lr_scheduler = None
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count())))

    criterion = nn.CrossEntropyLoss()
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_optim = copy.deepcopy(optimizer.state_dict())
    best_val_acc = 0.0

    model.to(device)
    if support_model is not None:
        support_model.to(device)
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
                
                x_pretrain = None
                if support_model is not None:
                    x_pretrain = support_model(inputs)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs, phase, x_pretrain, p_crossover)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        history['lr'].append(lr_scheduler.optimizer.param_groups[0]
                                             ["lr"]) if lr_scheduler else history['lr'].append(opt.lr)
                        if lr_scheduler is not None:
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
                    wandb.log({"train_acc": epoch_acc, "train_loss": epoch_loss}, step = epoch)
                    if lr_scheduler:
                        wandb.log(
                            {"lr": lr_scheduler.optimizer.param_groups[0]["lr"]}, step = epoch)
                    else:
                        wandb.log({"lr": opt.lr})
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                if opt.wandb_log:
                    wandb.log({"val_acc": epoch_acc, "val_loss": epoch_loss}, step = epoch)
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_optim = copy.deepcopy(optimizer.state_dict())
                    if opt.save_ckpt:
                        save_ckpt_(model, DIR_SAVE, "best.pt")

    time_elapsed = time.perf_counter() - since
    LOGGER.info(f"Training complete in \
                {time_elapsed // 3600}h \
                {time_elapsed % 3600 // 60}m \
                { time_elapsed % 60}s with \
                {num_epochs} epochs")
    LOGGER.info(f"Best val Acc: {round(best_val_acc.item(), 6)}")
    if opt.save_ckpt:
        save_ckpt_(model, DIR_SAVE, "last.pt")
    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_model_optim)

    if opt.log_result:
        plot_and_log_result(DIR_SAVE, history)
    if opt.save_ckpt:
        LOGGER.info(f"Best model weight saved at {DIR_SAVE}/weights/best.pt")
        LOGGER.info(f"Last model weight saved at {DIR_SAVE}/weights/last.pt")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='cuda device or cpu (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed will start at seed = 2 (default: %(default)s)')
    parser.add_argument('--base-init', type=str, default="He",
                        help='The method to initialize for parameters ["He", "Glorot"] (default: %(default)s)')
    # choices= ['N/A', 'maxVar', 'minVar', 'twoTailed', 'interLeaved', 'random']
    parser.add_argument('--keep', default='interLeaved', type=str,
                        help='Method to choose for down weight when using --transfer-weight flag')
    # choices= ['N/A', 'maxVar', 'minVar', 'twoTailed', 'interLeaved', 'random']
    parser.add_argument('--remove', default='random', type=str,
                        help='Method to choose for up weight when using --transfer-weight flag')
    # choices= ['zero', 'init']
    parser.add_argument('--type-pad', type=str, default="zero",
                        help='Add padding when up size kernel. \
                            if use init, it is base on the --base-init method. (default: %(default)s)')
    # choices= ['avg', 'max']
    parser.add_argument('--type-pool', type=str, default="avg",
                        help='down size kernel. Using when down size of kernel. (default: %(default)s)')
    # choices= ['CIFAR10', 'Intel', 'PetImages']
    parser.add_argument('--data-name', type=str, default="CIFAR10",
                        help='The name of dataset (default: %(default)s)')
    parser.add_argument('--data-root', type=str, default="./data",
                        help='Folder where the dataset is saved (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Mini-batch size for each iteration when training model (default: %(default)s)')
    parser.add_argument('--workers', type=int, default=2,
                        help='The number of worker for dataloader (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='The number of epochs in training (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for optimizer (default: %(default)s)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument('--lr-scheduler', action='store_true',
                        help='Learning rate scheduler during training. Default: MultiStepLR')
    parser.add_argument('--lr-step', nargs='+', type=float,
                        help='Lmilestones for learning rate scheduler (defaut: [0.7, 0.9])')
    parser.add_argument('--show-summary', action='store_true',
                        help='Show model summary with default input size (3, 224, 224)')
    parser.add_argument('--name', default='exp',
                        help='Project name will saved at runs/train/__name__')
    parser.add_argument('--save_ckpt', action='store_true',
                        help='Save model into checkpoint folder')
    parser.add_argument('--log-result', action='store_true',
                        help='Save result of training progress into checkpoint folder')
    parser.add_argument('--wandb-log', action='store_true',
                        help='Log result into WanDB')
    parser.add_argument('--wandb-name', type=str, default="wandb_log",
                        help='Log name in WanDB')
    opt = parser.parse_args()

    try:
        device_name = os.getlogin()
    except:
        device_name = "Colab/Cloud"
    if opt.wandb_log:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=opt.wandb_name,
            name=opt.name,
            tags=[device_name],
            config=vars(opt))
    else:
        wandb = None

    if opt.log_result:
        AppPath.RUN_DIR.mkdir(parents=True, exist_ok=True)
        AppPath.RUN_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    with open('./config/dataset.yaml') as f:
        dataset_config = yaml.load(f, Loader=SafeLoader)

    LOGGER.info(f"\n*** RUN ON SEED {opt.seed} ***")
    num_classes = dataset_config["dataset_config"][opt.data_name]["num_classes"]

    # Define the model before training
    #Source model
    source_model = torchmodel.vgg16(
        weights=torchmodel.VGG16_Weights.IMAGENET1K_V1)

    fc_mean, fc_std = [] ,[]
    with torch.no_grad():
        for fc in source_model.classifier:
            if isinstance(fc, nn.Linear):
                fc_mean.append(fc.weight.mean())
                fc_std.append(fc.weight.std())


    #Target mdoel
    target_model = CustomResnet._get_model_custom(model_base='resnet18', num_classes=num_classes)
    nn.init.normal_(
        target_model.fc.weight, 
        torch.Tensor(fc_mean).mean(), 
        torch.Tensor(fc_std).mean()
    )
    
    group_filter = [nn.Conv2d]

    var_transfer_config = {
        "type_pad": opt.type_pad,
        "type_pool": opt.type_pool,
        "choice_method": {
            "keep": opt.keep,
            "remove": opt.remove
        }
    }

    transfer_tool = TLV(
        standardization=FlattenStandardization(group_filter),
        matching=IndexMatching(),
        transfer=VarTransfer(**var_transfer_config)
    )  

    transfer_tool(
        from_module=source_model,
        to_module=target_model
    )
    # Finish define model

    train_loader, val_loader = get_train_valid_loader(
        dataset_name=opt.data_name,
        data_dir=opt.data_root,
        batch_size=opt.batch_size,
        augment=True,
        random_seed=opt.seed,
        num_workers=opt.workers
    )
    test_loader = get_test_loader(
        dataset_name=opt.data_name,
        data_dir=opt.data_root,
        batch_size=opt.batch_size,
        num_workers=opt.workers
    )
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    steps_per_epoch = len(dataloaders['train'])

    optimizer = torch.optim.AdamW(target_model.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)
    total_step = steps_per_epoch * opt.epochs
    if opt.lr_step is not None:
        milestones = [int(opt.lr_step[i] * total_step)
                      for i in range(len(opt.lr_step))]
    else:  # use default
        milestones = [int(0.6 * total_step), int(0.8 * total_step)]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestones)

    support_model = torchmodel.vgg16(weights = torchmodel.VGG16_Weights.IMAGENET1K_V1)
    # scale feature of vgg16 model from 25088 to 512
    support_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
    support_model.classifier = torch.nn.Identity()
    support_model.eval()


    if torch.cuda.device_count() > 1:
        targer_model = nn.DataParallel(target_model, 
                                       device_ids=list(range(torch.cuda.device_count())))

    best_model, val_acc = train_model(model=target_model,
                                      support_model=support_model,
                                      dataloaders=dataloaders,
                                      optimizer=optimizer,
                                      opt=opt,
                                      wandb=wandb,
                                      lr_scheduler=lr_scheduler)
    test_acc = test_model(best_model, dataloaders["test"], opt.device)
    LOGGER.info(f"Validation accuracy: {round(val_acc, 6)}")
    LOGGER.info(f"Test accuracy: {round(test_acc, 6)}")
    if opt.wandb_log:
        wandb.finish()
