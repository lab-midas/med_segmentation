import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import PIL
import scipy.io as sio


def save_histories_plot_images(history, dataset, config, mode='train_val',k_fold_index=0):
    """
    Save the history data from training and evaluating process
    :param history: the history object
    :param config:  type dict, config parameter
    :param dataset: type str, name of dataype
    :param mode:  type str: 'train_val' or 'predict'
    :param k_fold_index: type int:
    :return:
    """

    path_figures = config['result_rootdir'] + '/' + config[
        'model'] + '/figures/train_loss_and_metrics/' + dataset + '/'
    path_pickle = config['result_rootdir'] + '/' + config['model'] + '/train_history/' + dataset + '/'

    if not os.path.exists(path_figures): os.makedirs(path_figures)
    if not os.path.exists(path_pickle): os.makedirs(path_pickle)
    if history.history != {} and history.history is not None:
        if mode == 'train_val':
            plt.figure(1)
            y = history.history['loss']
            x1 = [i + 1 for i in range(len(y))]
            plt.plot(x1, y)
            y = history.history['val_loss']
            x2 = [(i + 1) * config['validation_freq'] for i in range(len(y))]
            plt.plot(x2, y)

            plt.title('Training Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train loss', 'Validate loss'], loc='upper right')
            plt.savefig(path_figures +config['model']+'_'+ dataset + '_' + str(k_fold_index) + '_loss.png')
            plt.savefig(path_figures +config['model']+'_'+ dataset + '_' + str(k_fold_index) + '_loss.pdf')
            plt.close(1)

            plt.figure(2)
            history_keys = history.history.keys()
            metric_list = [metric for metric in history_keys if 'loss' not in metric]
            for item in metric_list:
                plt.plot(history.history[item])
            plt.title('Traing Metric')
            plt.ylabel('Metrics')
            plt.xlabel('Epoch')
            plt.legend(metric_list, loc='upper right')
            plt.savefig(path_figures +config['model']+'_'+ dataset + '_' + str(k_fold_index) + '_metric.png')
            plt.savefig(path_figures+config['model']+'_' + dataset + '_' + str(k_fold_index) + '_metric.pdf')
            plt.close(2)

            history.history['validation_freq'] = config['validation_freq']
            with open(path_pickle +config['model']+'_'+ dataset + '_k_fold_' + str(k_fold_index) + '.pickle', 'wb') as fp:
                pickle.dump(history.history, fp, protocol=pickle.HIGHEST_PROTOCOL)
                print('Sucessfully save the ' + mode + ' loss and metrics of  ' + dataset + '.')
    else:
        print('Warning! No training / Validation loss or metric data! ')


def color_set(num_categories, costum_colormap=None, random_colormap=False, costum_max_value=255, costum_start=0,
              random_scale=1):
    """
    Generating color values by given number of categories.
    :param num_categories: type int: number of categories that need to be colorized
    :param costum_colormap: type list of list of 3 float: costume colormap
    :param random_colormap: type bool: True if random colors are chosen
    :param costum_max_value: type float: max value if default colormap is chosen
    :param costum_start: type int: start point of extraction from default colormap
    :param random_scale: type float in [0,1]: scale of random color if random colormap is chosen
    :return: color_map: type list of list of 3 float: generated color values
    """

    if costum_colormap is None and random_colormap == False:

        # Edit here if more default colors are needed
        default_colormap = (np.array([[0, 0, 0.8], [0, 0.8, 0], [0.8, 0, 0], [0.1, 0, 0.8], [0.3, 0.5, 0.1],
                                      [0, 0, 0.7], [0, 0.7, 0], [0.7, 0, 0], [0.1, 0, 0.7], [0.2, 0.5, 0.1],
                                      [0, 0, 0.7], [0, 0.7, 0], [0.7, 0, 0], [0.1, 0, 0.7], [0.2, 0.5, 0.1],
                                      [0, 0.1, 0.6], [0.5, 0.7, 0], [0.7, 0.1, 0], [0.2, 0.1, 0.7], [0.3, 0.4, 0.2],
                                      [0, 0.3, 0.7], [0, 0.7, 0.1], [0.4, 0.5, 0.3], [0.1, 0.2, 0.6], [0.2, 0.4, 0.3]])
                            * costum_max_value)
        assert (num_categories + costum_start <= len(default_colormap))
        return default_colormap[costum_start:num_categories + costum_start]

    elif costum_colormap is not None:
        return costum_colormap[:num_categories]

    elif random_colormap:
        colormap_set = [np.array([i, j, k]) * random_scale for k in np.arange(0, 255, 1) for j in np.arange(0, 255, 1)
                        for i in
                        np.arange(0, 255, 1)]
        corlor_map_index = np.random.randint(len(colormap_set), size=num_categories)
        color_map = np.array([colormap_set[i] for i in corlor_map_index])
        return color_map

    else:
        return None


def plot_mosaic(config, mask, slice_dim=2, colormap=None, vspace=2, hspace=2, col=5, rotate_k=None, flip_axis=None,
                origin_image=None, alpha_origin=0.3, dataset='0', name_ID='0',client_save_rootdir=None,image_type='predict'):
    """
 Plot the result of 3D category map or 3D grey image, layout in mosaic style.
    :param config: type dict: config parameter
    :param mask:  type ndarray: the 3D label or the predict data (category map, not one-hot).

    :param slice_dim: type int: dimension index of slicing.
            For 3D, 0 for Sagittal plain, 1 for Coronal plane, 2 for Axial plane.

    :param colormap: type list of list of 3 floats: color value of each category from :param: mask
    :param vspace: type int, vertical interval between each mosaic sub-figure. vspace=0 if no space between them.
    :param hspace: type int, horizontal interval between each mosaic sub-figure. vspace=0 if no space between them.
    :param col: type int: the column of mosaic figure
    :param origin_image: type ndarray: original 3D image (input image) with specified channel
    :param alpha_origin:   type float in [0,1]: transparency of :param: origin_image
    :param dataset: type str: name of the dataset
    :param name_ID: type str: name ID of the plotted image
    :return:

    """
    # Define variable h(height), w(width), and slices
    mask_shape = mask.shape[:3]
    mask = mask[:, :, :, 0]

    #if mask.shape != origin_image.shape:
    if mask_shape != origin_image.shape:
    ## sometimes after the unpatching, the predicted image is smaller than original
        difference = [(origin_image.shape[i] - mask_shape[i]) for i in range(len(origin_image.shape))]
        array_difference = [(0, difference[0]), (0, difference[1]), (0, difference[2])]
        new_mask = np.pad(mask, array_difference, mode='constant', constant_values=0)
        mask = new_mask
        print("new mask after padding: ", mask.shape)
        ##with this it is ensure that dimensions are good

    ## slice_dim = 2
    if slice_dim == 2:
        h, w, slices = mask_shape[0], mask_shape[1], mask_shape[2]
    elif slice_dim == 1:
        h, w, slices = mask_shape[0], mask_shape[2], mask_shape[1]
    else:
        h, w, slices = mask_shape[1], mask_shape[2], mask_shape[0]

    if rotate_k==1 or rotate_k==3 or rotate_k==-1 or rotate_k==-3:
        h,w=w,h
    ##_------------------------------------------------------------------

    num_category = len(np.unique(mask)) ## with Melanom Dataset should be 2 (tumor, not tumor)
    #art_mask = np.random.randint(5, size=mask.shape[:3])
    #art_mask = np.zeros(mask.shape[:3])
    #art_mask[100:, 100:, 150:] = 1
    #num_category = len(np.unique(art_mask))
    ##-----------------------------------------------------------------
    colormap = color_set(num_categories=num_category, costum_colormap=colormap)
    row = math.ceil(slices / col)
    # layout empty figure
    figure = Image.new('RGBA', (col * w + (col - 1) * vspace, row * h + (row - 1) * hspace))
    indice = [slice(None), slice(None), slice(None)]

    for row_index in range(row):
        for col_index in range(col):
            slice_index = row_index * col + col_index

            if slice_index < slices:
                indice[int(slice_dim)] = slice_index
                indice_ = tuple(indice)

                color_image = colormap[mask[indice_].astype(int) % num_category].astype('uint8')
                ##color_image = colormap[art_mask[indice_].astype(int) % num_category].astype('uint8')
                if rotate_k:
                    color_image= np.rot90(color_image, k=rotate_k)

                if flip_axis:
                    color_image=np.flip(color_image,axis=flip_axis)
                ##---------------------------------------------------------------------------------
                #art_mask = np.ones(mask.shape[:3])
                ##_----------------------------------
                ##---------------------------------------------------------------------------------
                im = Image.fromarray(color_image)
                im = im.convert("RGBA")

                if origin_image is not None:
                    origin_image_slice = origin_image[indice_]
                    origin_image = (
                                (origin_image - np.min(origin_image)) / (np.max(origin_image) + 1e-16) * 255).astype(
                        'uint8')
                    if rotate_k:
                        origin_image_slice= np.rot90(origin_image_slice, k=rotate_k)
                    if flip_axis:
                        origin_image_slice = np.rot90(origin_image_slice, k=rotate_k)

                    origin_image_slice = Image.fromarray(origin_image_slice).convert("RGBA")
                    #print("image shape: ", im.size)
                    #print("origin image slice", origin_image_slice.size)
                    #if im.size != origin_image_slice.size:
                        #print("mask size is: ", mask.shape)
                        #print("slice_",  indice_)
                        #print("color_image", color_image)

                    #im = Image.blend(im, origin_image_slice, alpha=alpha_origin)
                    im = Image.blend(im, origin_image_slice, alpha=0.0)

                # Draw slice index on the left top of the image
                #draw = ImageDraw.Draw(im)
                #font = ImageFont.truetype("arial.ttf", 14)
                #draw.text((7, 7), str(slice_index+1), font=font)
                figure.paste(im, (col_index * (w + vspace), row_index * (h + hspace)))

    dir_figures = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + '/figures/plot_mosaic/' + dataset + '/' + name_ID
    if client_save_rootdir is not None:
        dir_figures=client_save_rootdir+'/'+dir_figures
    if not os.path.exists(dir_figures): os.makedirs(dir_figures)
    path_figures = dir_figures + '/mosaic_'+image_type+'_' + config[
        'model'] + '_' + dataset + '_' + name_ID + '_' + 'slice_dim_' + str(slice_dim) + '.png'
    figure.save(path_figures)


def plot_area_ratio(config, list_images_series, slice_dim=2, merge_channel_plot=False, yscale='linear',
                    plot_label_series=None, dataset='data', figsize=None,plot_mean_only=False,client_save_rootdir=None):
    """
    Plot the mean and standard variation of the area ratio depending of slice of the dataset.
    The total slice number of each patient in the dataset can be varied.
    :param config: type dict: config parameter
    :param list_images_series: type list of list of one_hot ndarray: list of images series. This series can be : list of predict images,
            or list of labels.Each series must correspond to each other.
            The format: [ [predict_image_1, predict_image_2 ...], [label_image_1,label_image_2 ...] ...],
             each (predict or label)image must be 4 dimensional one_hot ndarray.

    :param slice_dim:  type int: dimension index of slicing.
            For 3D, 0 for Sagittal plane, 1 for Coronal plane, 2 for Axial plane.
    :param merge_channel_plot: type bool, True if plot all in one figure
    :param plot_label_series: type list of str: name of each series, default: ['predict', 'label']
    :param dataset: type str: name of this dataset
    :return:
    """

    for i in range(len(list_images_series)):
        # assert num of each sublist (number of patients) are same
        assert (len(list_images_series[0]) == len(list_images_series[i]))
    # num of patients
    sum_ = len(list_images_series[0])

    # shape of each patients 4D image
    #channel = list_images_series[0][0].shape[-1]
    channel = None

    mean_map_series = []
    std_map_series = []
    max_slice_series = []

    for list_img in list_images_series:  # for each sublist,
        sum_ratio = []
        max_slice = 0
        channel = list_img[0][0].shape[-1]

        for n in range(sum_):  # sum_: total number of patients in this series.
            img = list_img[n]
            shape = img.shape[:3]
            area, slice_ = shape[slice_dim - 2] * shape[slice_dim - 1], shape[slice_dim]
            max_slice = max(max_slice, slice_)  # calculate the max slice num from all the patients in this series.

            # Calculate area ratio
            #  Variable img_channels_area_ratio.shape:(slice, channel). Each pixel value: area ratio

            if slice_dim == 2:

                img_channels_area_ratio = np.array(
                    [[np.sum(img[:, :, i, j]) / float(area) * 100 for j in range(channel)] for i
                     in range(slice_)])
            elif slice_dim == 1:
                img_channels_area_ratio = np.array(
                    [[np.sum(img[:, i, :, j]) / float(area) * 100 for j in range(channel)] for i
                     in range(slice_)])
            else:
                img_channels_area_ratio = np.array(
                    [[np.sum(img[i, :, :, j]) / float(area) * 100 for j in range(channel)] for i
                     in range(slice_)])

            sum_ratio.append(img_channels_area_ratio)

        # Initialize sum map, fill with -1 (invalid data)
        # sum_map shape: (total num patient, total num of max slice, total num channel)
        sum_map = np.ones((sum_, max_slice, channel)) * -1
        # Fill the calculated data from above into the sum map
        for n in range(sum_):
            sum_map[n, 0:sum_ratio[n].shape[0], 0:sum_ratio[n].shape[1]] = sum_ratio[n]

        # Initialize mean map and standard variation map, fill with 0
        mean_map, std_map = np.zeros((max_slice, channel)), np.zeros((max_slice, channel))
        for sl in range(max_slice):
            for ch in range(channel):
                # Get area ratio data in the same slice index and the same channel index from all patients.
                sub_slice = sum_map[:, sl, ch]
                # Remove the invalid data
                sub_slice = sub_slice[sub_slice != -1]
                # Calculate mean (standard variation) and store them into the mean map (standard variation map).
                mean_map[sl, ch], std_map[sl, ch] = np.mean(sub_slice), np.std(sub_slice)

        mean_map_series.append(mean_map)
        std_map_series.append(std_map)
        max_slice_series.append(max_slice)

    if plot_label_series is None: plot_label_series = ['predict', 'label']

    if merge_channel_plot:  # Merge all series and all channels in one figure.
        if figsize is not None: figsize = tuple(figsize)
        plt.figure(1, figsize=figsize)
        plt.xlabel("Slice index")
        plt.ylabel("Average volumes percentage(%)")

        max_slice = max(max_slice_series)
        plt.xticks(list(range(0, max_slice + 1, 50)))
        indexes = list(range(0, max_slice))
        plt.ylim(*(0, 100))
        plt.yticks(np.linspace(0, 100, 11))
        plt.grid()
        line_style = ['-', '--', '-.', ':']
        for i in range(len(list_images_series)):

            colorset = color_set(num_categories=channel, costum_max_value=1, costum_start=i * channel)
            for ch in range(channel):
                # Plot standard variation gap
                if not plot_mean_only:
                    plt.fill_between(indexes, (mean_map_series[i][:, ch] - std_map_series[i][:, ch]),
                                     (mean_map_series[i][:, ch] + std_map_series[i][:, ch]),
                                     alpha=0.5, color=colorset[ch % channel])  # , color=colorset[ch % channel]

                # Plot mean line
                plt.plot(indexes, mean_map_series[i][:, ch], line_style[i % len(line_style)],
                         label=plot_label_series[i] + "_channel_" + str(ch),
                         color=colorset[ch % channel])  # , color=colorset[ch % channel]
        plt.yscale(yscale)
        plt.legend(loc="upper right")

        # Save figure
        path_dir = config['result_rootdir'] + '/' + config['model'] + '/figures/plot_area_ratio/' + dataset
        if client_save_rootdir is not None:
            path_dir = client_save_rootdir + '/' + path_dir
        if not os.path.exists(path_dir): os.makedirs(path_dir)
        path_figures = path_dir + '/area_ratio_merged_plot.png'

        plt.savefig(path_figures)
        plt.close(1)

    else:  # Merge the plots from different series in individual channel, save the figures by each channel,
        for ch in range(channel):
            plt.figure(ch)
            plt.xlabel("Slice index")
            plt.ylabel("Average volumes percentage(%)")
            max_slice = max(max_slice_series)
            plt.xticks(list(range(0, max_slice + 1, 50)))
            indexes = list(range(0, max_slice))
            plt.ylim(*(0, 100))
            plt.yticks(np.linspace(0, 100, 11))
            plt.grid()
            len_series = len(list_images_series)
            colorset = color_set(num_categories=len_series, costum_max_value=1, costum_start=ch * len_series)
            line_style = ['-', '--', '-.', ':']
            for i in range(len(list_images_series)):
                # Plot standard variation gap
                if not plot_mean_only:
                    plt.fill_between(indexes, (mean_map_series[i][:, ch] - std_map_series[i][:, ch]),
                                     (mean_map_series[i][:, ch] + std_map_series[i][:, ch]),
                                     alpha=0.5, color=colorset[i % len_series])

                # Plot mean line
                plt.plot(indexes, mean_map_series[i][:, ch], line_style[i % len(line_style)],
                         color=colorset[i % len_series],
                         label=plot_label_series[i] + "_channel_" + str(ch))
            plt.yscale(yscale)
            plt.legend(loc="upper right")

            # Save figure
            path_dir = config['result_rootdir'] + '/' + config['model'] + '/figures/plot_area_ratio/' + dataset
            if client_save_rootdir is not None:
                path_dir = client_save_rootdir + '/' + path_dir
            if not os.path.exists(path_dir): os.makedirs(path_dir)
            path_figures = path_dir + '/area_ratio_merged_plot_channel_' + str(ch) + '.png'
            plt.savefig(path_figures)
            plt.close(ch)




def plot_combine(image, heatmap, alpha=0.4, display=False, save_path=None,  verbose=False,
            rotate=False, pred_thresholds=[], real_thresholds=[]):
    """Still working..."""

    '''Combine image with heatmap, plot and save'''
    aspect = 0.1
    if len(real_thresholds) == 0:
        real_thresholds = np.zeros(pred_thresholds.shape)

    if rotate:
        image = np.rot90(image, k=1, axes=(1, 0))
        heatmap = np.rot90(heatmap, k=1, axes=(1, 0))
        aspect = 1.0 / aspect

    # Discrete color scheme
    cMap = ListedColormap(['midnightblue', 'darkslateblue', 'darkcyan', 'olive', 'goldenrod'])

    # Display
    fig, ax = plt.subplots()
    image = ax.pcolormesh(image)
    heatmap = ax.pcolor(heatmap, alpha=alpha, cmap=cMap)
    ax.set_aspect(aspect=aspect)

    cbar = plt.colorbar(heatmap)
    landmark_dict = {0: 'Wrists', 1: 'Shoulders', 2: 'Liver_dome', 3: 'Hips', 4: 'Heels', 5: 'Below'}
    cbar.ax.set_ylabel('Class')
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(landmark_dict):
        cbar.ax.text(.5, (2 * j + 1) / 10.0, landmark_dict[j], ha='center', va='center', rotation=90)
    cbar.ax.get_yaxis().labelpad = 5
    cbar.ax.invert_yaxis()

    real_thresholds = np.flip(real_thresholds, 0)
    for t in range(pred_thresholds.size):
        plt.axhline(y=pred_thresholds[t], color='r', ls='dashed', label='Predicted Threshold')
        plt.axhline(y=int(real_thresholds[t]), color='w', ls='dotted', label='Real Threshold')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:2], labels=['Predicted Threshold', 'Real Threshold'], loc='upper center',
              bbox_to_anchor=(0.5, -0.1), ncol=2)
    #    ax.legend.get_frame().set_facecolor('0.5')

    plt.title('Decision Map')
    plt.ylabel('HF Slice Number')
    plt.xlabel('LR Slice Number')

    zoom = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)

    if display:
        plt.show()

    if save_path is not None:
        if verbose:
            print('Heatmap with image saved at ', save_path)
        save_name = save_path + '/heatmap_' + str(alpha)
        plt.savefig(save_name + '.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(save_name + '.svg', bbox_inches='tight', pad_inches=0)


def plot_by_slice(config, masks_list, slice_, channel_only=None, slice_dim=2, space=1, title_list=None,
                          origin_image=None, alpha_origin=0.6, figure_layout='c',
                          colormap=None, name_ID='0', dataset='data',client_save_rootdir=None):
    """
    Plot the image mask at the specific slice.
    :param config: type dict: config parameter
    :param masks_list: type list of ndarray: the label or the predict data (category map, not one-hot). The format can
                    either be [ndarray of predict, ndarray of label...], or [ndarray of predict], or [ndarray of label]
    :param slice_: type int: slice index of displaying image
    :param channel: type int: channel index of displaying image
    :param slice_dim: type int: dimension index of slicing.
            For 3D, 0 for Sagittal plain, 1 for Coronal plane, 2 for Axial plane.
    :param title_list: type list of str: title of the images
    :param origin_image: type ndarray: original image (input image)
    :param figure_layout  type str: layout of subplots, 'c' for column and 'r' for row.
    :param alpha_origin:  type float in [0,1]: transparency of :param: origin_image
    :param alpha_mask:  type float in [0,1]: transparency of :param: masks_list
    :param colormap: type list of list of 3 floats: color value of each category from :param: mask_list
    :param name_ID: type str: name ID of the plotted image
    :param dataset: type str: name of the dataset
    :param space: type int, space between the images
    :return:
    """

    if not isinstance(masks_list, list):
        masks_list = [masks_list]
    num_masks = len(masks_list) #


    # if choose to show the specific channel 'channel_only'
    if channel_only is not None:
        for im in masks_list:
            im[im != channel_only] = 0

    # Setting slicing parameter
    indice = [slice(None), slice(None), slice(None)]
    indice[int(slice_dim)] = slice_ # choose 0 for Sagittal plane, 1 for Coronal plane, 2 for Axial plane
    indice = tuple(indice)
    # Change origin image into type np.uint8
    # Slice the mask list at the specified position.
    masks_list = [im[indice] for im in masks_list]
    h, w = masks_list[0].shape[0:2]
    if figure_layout == 'c': # Plots are column arraged
        # Set new empty image with specific size.
        figure = Image.new('RGBA', (num_masks * w + (num_masks - 1) * space, h * 1))
    else: # Plots are row arraged
        # Set new empty image with specific size.
        figure = Image.new('RGBA', (w * 1, num_masks * h + (h - 1) * space))
    # choose and process the  origin image as background
    origin_image_slice = None #
    if origin_image is not None:
        origin_image_slice = origin_image[indice]
        origin_image_slice = ((origin_image_slice - np.min(origin_image_slice)) / (
                    np.max(origin_image_slice) + 1e-16) * 255).astype('uint8')
        origin_image_slice = Image.fromarray(origin_image_slice).convert('RGBA')

    for image_index in range(0, num_masks):
        mask = masks_list[image_index].astype(int)

        # Colorize the masks by function color_set
        mask_categories = len(np.unique(mask))
        colormap = color_set(num_categories=mask_categories + 1, costum_colormap=colormap)  # use default colorset

        # Color the image
        color_image = colormap[mask % mask_categories].astype('uint8')
        im = Image.fromarray(color_image).convert("RGBA")

        # Merge the origin image with label/predict image
        if origin_image is not None:
            im = Image.blend(im, origin_image_slice, alpha=alpha_origin)

        # Show title on the top
        if title_list is not None:
            draw = ImageDraw.Draw(im)
            #font = ImageFont.truetype("arial.ttf", 10)
            #draw.text((2, 2), title_list[image_index], font=font)

        if figure_layout == 'c':
            figure.paste(im, (image_index * (w + space), 0))
        else:
            figure.paste(im, (0, image_index * (h + space)))

    # Save figure
    dir_figures = config['result_rootdir'] + '/' + config[
        'model'] + '/figures/plot_by_slice/' + dataset + '/' + name_ID
    if client_save_rootdir is not None:
        dir_figures = client_save_rootdir + '/' + dir_figures
    if not os.path.exists(dir_figures): os.makedirs(dir_figures)
    path_figures = dir_figures + '/profile_' + config['model'] + '_' + name_ID + '_plane_' + str(
        slice_dim) + '_slice_' + str(slice_) + '.png'
    figure.save(path_figures)
    return
