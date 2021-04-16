import h5py
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib
from models.loss_function import *
from models.metrics import *
from util import *

#from scipy.ndimage import label

from scipy import ndimage

from med_io.postprocessing import *
from models.loss_torch import *

def save_dict_image(config, dataset, name_ID, dict):
    save_validation_data_dir = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
                   '/validation_result/' + dataset + '/' + name_ID

    if not os.path.exists(save_validation_data_dir): os.makedirs(save_validation_data_dir)
    save_path = save_validation_data_dir + '/' + 'validation_' + config[
        'model'] + '_' + dataset + '_' + name_ID + '.mat'
    sio.savemat(save_path, dict)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    #C = tensor.size(1)
    C = 2
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-5,
                             ignore_index=None, weight=None):
    # assumes that input is a normalized probability
    # input and target shapes must match
    #assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)

def save_csv(config, dataset, dict_info):
    #csv_path = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
    #               '/validation_result/csv_data/' + dataset + '/'

    csv_path = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
               '/validation_result/csv_data_all/' + dataset + '/'

    if not os.path.exists(csv_path): os.makedirs(csv_path)

    csv_name = 'validation.csv'

    df = pd.DataFrame.from_dict(dict_info)
    df.to_csv(csv_path+csv_name)

def validation(config, datasets):
    """Validate lesion segmentation (use manual labels as reference).
    Args:
        config: configuration file from application.
    """
    # copy over
    voxel_vol = config['voxel_vol']
    petsuv_scale = config['petsuv_scale']
    project = config['project']


    for dataset in datasets:


        result_dict = {}

        ids_imgs, _ = get_test_keys(config, dataset)

        ##########------------------------------------------------------
        #path_imgs_predict = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
        #               '/predict_result/' + dataset
        #ids_imgs = os.listdir(path_imgs_predict)

        #############-----------------------------------------------------
        #path_to_predictions = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
        #                    '/postprocessing_result/' + dataset
        path_to_postprocess = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
                        '/postprocessing_result_all/' + dataset

        rootdir_file = config['rootdir_raw_data_img'][dataset]
        Data_Reader = HD5F_Reader(dataset, rootdir_file)

        img_IDs = Data_Reader.img_IDs
        file_keys = Data_Reader.file_keys
        #ids_imgs = [ids_imgs[5]]
        results_validation = []

        png_dir = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
                  '/figures/plot_validation/' + dataset

        i = 0


        for id_img in ids_imgs:

            print("validating image index: " + str(i) + "........................")
            png_dir_img = png_dir + '/' + id_img + '/'

            if not os.path.exists(png_dir_img): os.makedirs(png_dir_img)

            ## mask without postprocessing
            #_, mask_predicted, _ = read_mat_file(
            #    path=(path_to_predictions + '/' + id_img + '/predict_' + config['model'] +
            #          '_' + dataset + '_' + id_img + '.mat'), read_info=False, read_img=False,
            #    labels_name=['predict_image'])

            ## mask with postprocessing
            _, mask_predicted, _ = read_mat_file(
                path=(path_to_postprocess + '/' + id_img + '/postprocess_' + config['model'] +
                      '_' + dataset + '_' + id_img + '.mat'), read_info=False, read_img=False,
                labels_name=['postprocess_image'])

            print("values mask predicted: ", np.unique(mask_predicted, return_counts=True))

            img_h5 = Data_Reader.file[file_keys[0]][id_img]
            img_array = np.rollaxis(np.float32(np.array(img_h5)), 0, 4)
            pet_img = img_array[..., 0].astype(np.float32)
            mask_h5 = Data_Reader.file[file_keys[2]][id_img]
            mask_array_original = np.rollaxis(np.float32(np.array(mask_h5)), 0, 4)
            mask = mask_array_original#[..., 0]
            print("values mask original: ", np.unique(mask, return_counts=True))

            # store images and masks in mat files
            #dict_info = {'image': img_array,
            #             'mask_original': mask,
            #             'mask_predicted': mask_predicted}
            #save_dict_image(config, dataset=dataset, name_ID=id_img, dict=dict_info)

            # evaluate dice between predicted mask and original mask
            oh_mask = convert_integers_to_onehot(mask, num_classes=2)
            print("values mask original one hot: ", np.unique(oh_mask[..., 0], return_counts=True))
            print("values mask original one hot: ", np.unique(oh_mask[..., 1], return_counts=True))

            oh_mask_predicted = convert_integers_to_onehot(mask_predicted, num_classes=2)
            print("values mask predicted one hot: ", np.unique(oh_mask_predicted[..., 0], return_counts=True))
            print("values mask predicted one hot: ", np.unique(oh_mask_predicted[..., 1], return_counts=True))

            dice_loss = dice_loss_melanoma(oh_mask, oh_mask_predicted, config)
            metric_dice = Metric()
            dice_0 = metric_dice.dice_coef_per_class(0, oh_mask, oh_mask_predicted, config)
            dice_1 = metric_dice.dice_coef_per_class(1, oh_mask, oh_mask_predicted, config)
            dice_mean = metric_dice.dice_coef_all(oh_mask, oh_mask_predicted, config)

            #y_o = np.rollaxis(np.expand_dims(oh_mask, axis=0), axis=4, start=1)
            #y_p = np.rollaxis(np.expand_dims(oh_mask_predicted, axis=0), axis=4, start=1)
            #dice_torch_loss, dice_torch = DiceLoss(y_p, y_o)
            #dice_torch = compute_per_channel_dice(y_p, y_o)


            # evaluate jaccard between predicted mask and original mask
            #jaccard = jaccard_dist_loss(oh_mask, oh_mask_predicted, config)

            # evaluate MSE between predicted mask and original mask
            #mse = tf.keras.losses.mean_squared_error(oh_mask, oh_mask_predicted)
            #mse = mse.numpy()

            # volumina
            volume_manual = np.sum(mask) * voxel_vol
            volume_prediction = np.sum(mask_predicted) * voxel_vol

            # calculate uptake and number of lesions
            uptake_manual = np.ma.array(mask[..., 0] * pet_img, mask=mask[..., 0] == 0) * petsuv_scale
            uptake_prediction = np.ma.array(mask_predicted[..., 0] * pet_img, mask=mask_predicted[..., 0] == 0) * petsuv_scale
            print("values mask original: ", np.unique(mask, return_counts=True))
            print("values mask predicted: ", np.unique(mask_predicted, return_counts=True))

            #mask_ui = np.uint8(mask[..., 0])
            #mask_predicted_ui = np.uint8(mask_predicted[..., 0])
            #print("values mask original: ", np.unique(mask_ui, return_counts=True))
            #print("values mask predicted: ", np.unique(mask_predicted_ui, return_counts=True))

            _, num_lesions_manual = ndimage.label(mask[..., 0])
            _, num_lesions_prediction = ndimage.label(mask_predicted[..., 0])

            # plot validation figures
            fig, ax = plt.subplots(2, 1, figsize=[10, 15])
            ax[0].title.set_text('ground truth')
            ax[0].imshow(pet_img.max(axis=1), cmap='gray', vmin=0, vmax=0.2)
            ax[0].axis('off')
            mip = mask.max(axis=1)
            mip = np.ma.array(mip, mask=(mip == 0))
            ax[0].imshow(mip, alpha=0.8, vmin=0, cmap='coolwarm')

            ax[1].title.set_text(f'prediction \n DICE bg {dice_0: .3f} lesion {dice_1: .3f} project {project}')
            ax[1].axis('off')
            ax[1].imshow(pet_img.max(axis=1), cmap='gray', vmin=0, vmax=0.2)
            mip = mask_predicted.max(axis=1)
            mip = np.ma.array(mip, mask=(mip == 0))
            ax[1].imshow(mip, alpha=0.8, vmin=0, cmap='coolwarm')
            plt.tight_layout()
            name_file = 'validation_' + id_img + '.png'
            plt.savefig(png_dir_img + name_file, bbox_inches='tight', pad_inches=0)
            #plt.show()
            plt.close(fig)

            result = [id_img, config['project'],
                      volume_manual, volume_prediction,
                      dice_0.numpy(), dice_1.numpy(), dice_mean.numpy(),
                      uptake_manual.mean(),
                      uptake_manual.std(),
                      np.ma.median(uptake_manual.mean()),
                      uptake_manual.min(),
                      uptake_manual.max(),
                      num_lesions_manual,
                      uptake_prediction.mean(),
                      uptake_prediction.std(),
                      np.ma.median(uptake_prediction.mean()),
                      uptake_prediction.min(),
                      uptake_prediction.max(),
                      num_lesions_prediction]
            print(result)
            results_validation.append(result)
            i=i+1


        results = list(zip(*results_validation))

        # save results
        result_dict['key'] = results[0]
        result_dict['project'] = results[1]
        result_dict['volume_manual'] = results[2]
        result_dict['volume_prediction'] = results[3]
        result_dict['dice_background'] = results[4]
        result_dict['dice_lesion'] = results[5]
        result_dict['dice_total'] = results[6]
        result_dict['uptake_mean_manual'] = results[7]
        result_dict['uptake_std_manual'] = results[8]
        result_dict['uptake_median_manual'] = results[9]
        result_dict['uptake_min_manual'] = results[10]
        result_dict['uptake_max_manual'] = results[11]
        result_dict['num_lesions_manual'] = results[12]
        result_dict['uptake_mean_prediction'] = results[13]
        result_dict['uptake_std_prediction'] = results[14]
        result_dict['uptake_median_prediction'] = results[15]
        result_dict['uptake_min_prediction'] = results[16]
        result_dict['uptake_max_prediction'] = results[17]
        result_dict['num_lesions_prediction'] = results[18]

        # csv dice results to csv
        save_csv(config, dataset, result_dict)
