import os

import numpy as np
import nibabel as nib

from PIL import Image


def plot_mosaic(prediction, raw_image, path_figures, slice_dim=2, vspace=2, hspace=2, num_cols=5, alpha_raw=0.6,
                rotate=1, flip_axis=None):
    """Adapted version of med_segmentation mosaic plot

    Args:
        prediction (np.array): predicted segmentation map
        raw_image (np.array): one channel of raw image
        path_figures (str): path of figure, including patient identifier
        slice_dim (int): slice array dimension
        vspace (int): layout of figure, vertical spacing
        hspace (int): layout of figure, horizontal spacing
        num_cols (int): layout of figure, number of columns
        alpha_raw (float): layout of figure, alpha value of raw image
        rotate (int): k value of rotation
        flip_axis (int): flip axis of image
    """
    # Define variable h(height), w(width), and slices
    colormap = [[0.0, 0, 0.1], [0, 0.9, 0.3], [0.9, 0.7, 0.1], [0.9, 0.1, 0.1], [0.2, 0.1, 0.8], [0.8, 0.2, 0.1]]
    colormap = (np.array(colormap) * 255).astype('int32')
    prediction_shape = prediction.shape[:3]
    if slice_dim == 2:
        h, w, num_slices = prediction_shape[0], prediction_shape[1], prediction_shape[2]
    elif slice_dim == 1:
        h, w, num_slices = prediction_shape[0], prediction_shape[2], prediction_shape[1]
    else:
        h, w, num_slices = prediction_shape[1], prediction_shape[2], prediction_shape[0]
    if rotate in [-3, -1, 1, 3]:
        h, w = w, h
    num_classes = len(np.unique(prediction))
    colormap = colormap[:num_classes]
    num_rows = int(np.ceil(num_slices/num_cols))
    # layout empty figure
    figure = Image.new('RGBA', (num_cols * w + (num_cols - 1) * vspace, num_rows * h + (num_rows - 1) * hspace))
    index = [slice(None), slice(None), slice(None)]
    for row_index in range(num_rows):
        for col_index in range(num_cols):
            slice_index = row_index * num_cols + col_index
            if slice_index < num_slices:
                index[int(slice_dim)] = slice_index
                index_ = tuple(index)
                color_image = colormap[prediction[index_].astype(int) % num_classes].astype('uint8')
                if rotate:
                    color_image = np.rot90(color_image, k=rotate)
                if flip_axis:
                    color_image = np.flip(color_image, axis=flip_axis)
                im = Image.fromarray(color_image)
                im = im.convert("RGBA")
                if raw_image is not None:
                    raw_image_slice = raw_image[index_]
                    raw_image = ((raw_image - np.min(raw_image)) / (np.max(raw_image) + 1e-16) * 255).astype('uint8')
                    if rotate:
                        raw_image_slice = np.rot90(raw_image_slice, k=rotate)
                    if flip_axis:
                        raw_image_slice = np.flip(raw_image_slice, axis=flip_axis)
                    raw_image_slice = Image.fromarray(raw_image_slice).convert("RGBA")
                    im = Image.blend(im, raw_image_slice, alpha=alpha_raw)
                figure.paste(im, (col_index * (w + vspace), row_index * (h + hspace)))
    figure.save(path_figures)


path_results_nnUNet = '/mnt/share/rahauei1/nnUNet_trained_models/nnUNet/3d_fullres/Task001_AdiposeTissue/'\
                      + 'nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw'
path_raw_data = '/mnt/share/rahauei1/nnUNet_raw_data/Task001_AdiposeTissue/imagesTr'
path_dir_save = '/mnt/share/rahauei1/Results/nnUNet_visualization'
validation_images = sorted([i for i in os.listdir(path_results_nnUNet) if i.endswith('.nii.gz')])
fat_images = [i for i in os.listdir(path_raw_data) if '_0000.nii.gz' in i]
raw_images = []
for i in validation_images:
    raw_images.append(*[j for j in fat_images if i.split('.')[0] in j])
raw_images = sorted(raw_images)
pred = nib.load(os.path.join(path_results_nnUNet, validation_images[0]))
pred = np.array(pred.get_fdata())
raw = nib.load(os.path.join(path_raw_data, raw_images[0]))
raw = np.array(raw.get_fdata())
path_save = os.path.join(path_dir_save, validation_images[0].split('.')[0] + '.png')
plot_mosaic(pred, raw, path_save)
