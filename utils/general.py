import copy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import platform
import pickle


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_and_log_result(path_save, history):

    hist = os.path.join(path_save, "history.pkl")
    with open(hist, 'wb') as f:
        pickle.dump(history, f)

    train_loss_hist = np.array(history["train_loss"])
    train_acc_hist = np.array(history["train_acc"])
    val_loss_hist = np.array(history["val_loss"])
    val_acc_hist = np.array(history["val_acc"])
    lr_hist = np.array(history["lr"])
    epochs = np.arange(len(train_loss_hist))

    train_result = os.path.join(path_save, "train_result.txt")
    with open(train_result, "w") as f:
        f.write("Epoch\tLoss\tAccuracy\n")
        for epoch, loss, acc in zip(epochs, np.round(train_loss_hist, 6), np.round(train_acc_hist, 6)):
            f.write(f"{epoch}\t{loss}\t{acc}\n")
    
    val_result = os.path.join(path_save, "val_result.txt")
    with open(val_result, "w") as f:
        f.write("Epoch\tLoss\tAccuracy\n")
        for epoch, loss, acc in zip(epochs, np.round(val_loss_hist), np.round(val_acc_hist, 6)):
            f.write(f"{epoch}\t{loss}\t{acc}\n")

    epochs = np.arange(0, train_loss_hist.shape[0], 1, dtype=int)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Training result')

    axs[0, 0].plot(epochs, train_loss_hist)
    axs[0, 0].set_title("train loss")
    axs[0, 0].set_ylabel("loss")

    axs[0, 1].plot(epochs, train_acc_hist)
    axs[0, 1].set_title("train acc")
    axs[0, 1].set_ylabel("acc")

    axs[1, 0].plot(epochs, val_loss_hist)
    axs[1, 0].set_title("val loss")
    axs[1, 0].set_ylabel("loss")

    axs[1, 1].plot(epochs, val_acc_hist)
    axs[1, 1].set_title("val acc")
    axs[1, 1].set_ylabel("acc")


    for ax in fig.get_axes():
        ax.set_xlabel("Epochs")
        # ax.set_xticks(epochs)

    fig.tight_layout()
    fig.savefig(f"{path_save}/result.png")
    plt.close(fig) 

    steps = np.arange(lr_hist.shape[0])
    fig_lr, ax_lr = plt.subplots(nrows=1, ncols=1)
    ax_lr.plot(steps, lr_hist)
    ax_lr.set_title('Training learning rate')
    fig_lr.savefig(f"{path_save}/result_lr.png")
    plt.close(fig_lr) 

def save_ckpt_(model, PATH, name_ckpt):
    path_save = os.path.join(PATH, "weights")

    os.makedirs(path_save, exist_ok=True)
    
    model_ckpt = copy.deepcopy(model.state_dict())

    models_ckpt = {
        "model_state_dict": model_ckpt,
    }

    torch.save(models_ckpt,os.path.join(path_save, name_ckpt))


def load_ckpt(ckpt_path, model, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0]) 
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


