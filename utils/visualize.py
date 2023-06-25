import matplotlib.pyplot as plt
import copy

def get_feature_map(model, conv_idx, image):
    temp_model = copy.deepcopy(model)
    model_idx = temp_model.features[:conv_idx + 1]
    features = model_idx(image).cpu().detach().numpy()
    return features


def get_weight_conv(kernels, conv_idx):
    return kernels[str(conv_idx)]


def get_all_kernel(model):
    kernels = {}
    for idx, module in model.features._modules.items():
        if module.__class__.__name__ == 'Conv2d':
            filters = module.weight.cpu().detach().numpy()
            f_min, f_max = filters.min(), filters.max()
            filters = ((filters - f_min) / (f_max - f_min))* 255
            kernels[idx] = filters
    
    return kernels


def visualize_filter(feature_map, kernels, num_kernel, num_ch_Ofkernel, show_fig = True, save_fig = False, title = 'visualize', path_save = '/', transparent = False):

    if num_kernel > len(kernels):
        raise Exception("num_kernel must <= {}".format(len(kernels)))
    
    if num_ch_Ofkernel > len(kernels[0]):
        raise Exception("num_ch_Ofkernel must <= {}".format(len(kernels[0])))
    
    row = num_kernel
    col = num_ch_Ofkernel + 1

    fig, axs = plt.subplots(row, col)
    fig.suptitle(title, fontsize = 40)
    fig.set_figheight(3 * row)
    fig.set_figwidth(3 * col)

    for r in range(row):
        kernel = kernels[r, :, :, :]
        for c in range(col - 1):
            axs[r,c].imshow(kernel[c, :, :])
        axs[r,(c+1)].imshow(feature_map[r, :, :])

    fig.tight_layout()
    if show_fig:
        plt.show()

    if save_fig:
        fig.savefig(path_save, facecolor = 'white', transparent = transparent, dpi = 160)
    plt.close(fig)




