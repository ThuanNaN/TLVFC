import os
import wandb
import argparse
from utils.trainer import train_model, test_model
from utils.general import seed_everything, AppPath
from dataset_loader.dataset import get_train_valid_loader, get_test_loader
from dataset_loader.dataset_utils import get_mean_and_std
import yaml
from yaml.loader import SafeLoader

seed_everything(2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-try', type=int, default=1, 
                        help='The number of experiments (default: %(default)s)')
    
    parser.add_argument('--device', default='cuda', choices = ['cuda', 'cpu'], \
                        help='cuda device or cpu (default: %(default)s)')
    
    parser.add_argument('--seed', type=int, default=2, 
                        help='random seed will start at seed = 2 (default: %(default)s)')
    
    parser.add_argument('--model-group', type=str, default='resnet', \
                        choices= ['vgg', 'resnet'],\
                        help='The group of model. ["vgg", "resnet"] (default: %(default)s)')
    
    parser.add_argument('--model-name', type=str, default='vgg16_5x5_Down',
                        choices= ['vgg16_5x5_Down', 'vgg16_5x5_Up', 'vgg16_5x5_DownUp', 'vgg16_5x5_Sort', 'vgg16_5x5_Long',  \
                                  'vgg16', 'vgg19', 'resnet18', 'resnet34'],\
                        help='The type of initialization model. ["resnet18", "resnet34", "vgg16", "vgg19", \
                            "vgg16_5x5_Down", "vgg16_5x5_Up", "vgg16_5x5_DownUp"] (default: %(default)s)')

    parser.add_argument('--pretrain-group', type=str, default='vgg', \
                        choices= ['vgg', 'resnet'],\
                        help='The group of pretrain weight. ["vgg", "resnet"] (default: %(default)s)')
    
   
    parser.add_argument('--pretrain-name', type=str, default='vgg16',
                        help='The weight name of pretrain model. \
                            ["vgg16", "vgg19", "resnet18", "resnet34"] (default: %(default)s)')
    
    parser.add_argument('--base-init', type=str, default="He", required=True,
                        help='The method to initialize for parameters ["He", "Glorot", "Trunc"] (default: %(default)s)')
    
    parser.add_argument('--transfer-weight', action='store_true', 
                        help='Using weight transfer method from pre-trained. \
                            If not, using method in flag --base-init for all parameters')
    
    parser.add_argument('--keep', default='interLeaved', type=str, 
                        # choices= ['N/A', 'maxVar', 'minVar', 'twoTailed', 'interLeaved', 'random'], 
                        help='Method to choose for down weight when using --transfer-weight flag')
    
    parser.add_argument('--remove', default='random', type=str,
                        # choices= ['N/A', 'maxVar', 'minVar', 'twoTailed', 'interLeaved', 'random'], 
                        help='Method to choose for up weight when using --transfer-weight flag')
    
    parser.add_argument('--type-pad', type=str, default="zero", 
                        # choices= ['zero', 'init'],
                        help='Add padding when up size kernel ["zero", "init"]. \
                            if use init, it is base on the --base-init method. (default: %(default)s)')
    
    parser.add_argument('--type-pool', type=str, default="avg", 
                        # choices= ['avg', 'max'],
                        help='down size kernel ["avg", "max"]. Using when down size of kernel. (default: %(default)s)')
    
    parser.add_argument('--num-candidate', type=int, default=3, 
                        help='The number of cadidate nearest to matching weight (default: %(default)s)')
    
    parser.add_argument('--cand-select-method', type=str, default="max", choices= ['max', 'min'],
                        help='The procedure for selecting a candidate list. Definitely in ["max" and "min"] (default: %(default)s)')
    
    parser.add_argument('--mapping-type', type=str, default="relative", choices= ['relative', 'absolute'],
                        help='Method to mapping the index of convolution layer from ckpt to model. \
                            Definitely in ["relative" and "absolute"] (default: %(default)s)')
    
    parser.add_argument('--data-name', type=str, default="CIFAR10", 
                        help='The name of dataset. It is used to define some arguments in dataloader (default: %(default)s)')
    
    parser.add_argument('--data-root', type=str, default="./data", 
                        help='Folder where the dataset is saved (default: %(default)s)')
    
    parser.add_argument('--batch-size', type=int, default=256, 
                        help='Mini-batch size for each iteration when training model (default: %(default)s)')
    
    parser.add_argument('--workers', type=int, default=2, 
                        help='The number of worker for dataloader (default: %(default)s)')

    parser.add_argument('--epochs', type=int, default=50, 
                        help='The number of epochs in training (default: %(default)s)')
    
    parser.add_argument('--adam', action='store_true', 
                        help='Use Adam optimier. If not, using SGD is a optimizer')
    
    parser.add_argument('--adamW', action='store_true', 
                        help='Use AdamW optimier. If not, using SGD is a optimizer')
    
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate for optimizer (default: %(default)s)')
    
    parser.add_argument('--weight-decay', type=float, default=2e-8, 
                        help='Weight decay for optimizer (default: %(default)s)')
    
    parser.add_argument('--lr-scheduler', action='store_true', 
                        help='Learning rate scheduler during training. Default: torch.optim.lr_scheduler.MultiStepLR')
    
    parser.add_argument('--lr-step', nargs='+', type=float,  
                        help='Lmilestones for learning rate scheduler (defaut: [0.7, 0.9])')
    
    parser.add_argument('--warm-up', action='store_true', 
                        help='Scheduler warmup with learning rate in start training')
    
    parser.add_argument('--num-warm-up', type=int, default=10, 
                        help='The number of epochs for warmup scheduler')

    parser.add_argument('--show-summary', action='store_true', 
                        help='Show model summary with default input size id (3, 224, 224)')
    
    parser.add_argument('--name', default='exp', 
                        help='Project name will saved at runs/train/__name__')
    
    parser.add_argument('--save_ckpt', action='store_true' , 
                        help='Save model into checkpoint folder')
    
    parser.add_argument('--log-result', action='store_true' , 
                        help='Save result of training progress into checkpoint folder')
    
    parser.add_argument('--wandb-log', action='store_true' , 
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
            group=opt.model_name, job_type="train", name=opt.name,
            tags=[device_name],
            config=vars(opt)
        )
    else:
        wandb = None

    if opt.log_result:
        AppPath.RUN_DIR.mkdir(parents=True, exist_ok=True)
        AppPath.RUN_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    with open('./config/dataset.yaml') as f:
        dataset_config = yaml.load(f, Loader=SafeLoader)


    lst_val_acc = []
    lst_test_acc = []
    for i in range(opt.num_try):
        print(f"\n*** RUN ON SEED {opt.seed} ***")

        opt.num_classes = dataset_config["dataset_config"][opt.data_name]["num_classes"]
        train_loader, val_loader = get_train_valid_loader(
            dataset_name=opt.data_name,
            data_dir=opt.data_root,
            batch_size = opt.batch_size,
            augment = True,
            random_seed = opt.seed,
            num_workers = opt.workers
        )

        test_loader = get_test_loader(
            dataset_name=opt.data_name,
            data_dir=opt.data_root,
            batch_size = opt.batch_size,
            num_workers = opt.workers
        )

        dataloaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        model, val_acc = train_model(opt = opt, dataloaders = dataloaders, wandb = wandb)

        test_acc = test_model(model, dataloaders["test"], opt.device)
        lst_val_acc.append(val_acc)
        lst_test_acc.append(test_acc)

        opt.seed += 1

    print("Mean_val_acc: ", round(sum(lst_val_acc) / len(lst_val_acc), 6))
    print("Mean_test_acc: ", round(sum(lst_test_acc) / len(lst_test_acc), 6))

    if opt.wandb_log:
        wandb.finish()