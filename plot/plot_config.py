from .plot_figure import *
import numpy as np


def plot_figures_single(config, dict_data, dataset='0', name_ID='0', client_save_rootdir=None):
    """
    Plot figure configuration of a single patient.
    :param config: type dict: config parameter
    :param dict_data: type dict of ndarray: plot data
    :param dataset: type str: name of the dataset
    :param name_ID: type str: name ID of the patient
    :param client_save_rootdir: type str: root dir of saving plot files. None if use default directory
                                which is defined in the yaml.
    :return:
    """
    # No plot figure in config file
    if config['plot_figure'] == [] or config['plot_figure'] == {} or config['plot_figure'] is None:
        return
    # config 'plot figure', str->list
    elif isinstance(config['plot_figure'], str):
        config['plot_figure'] = [config['plot_figure']]

    for figure_name in config['plot_figure']: # figure_name can be plot_mosaic, plot_by_slice,plot_area_ratio

        img_data = dict_data['original_image']
        if figure_name == 'plot_mosaic':
            # mosaic plot settings
            if config['colormap'] is not None:
                config['plot_mosaic']['colormap'] = (np.array(config['colormap']) * 255).astype('int32')
            if config['display_origin_image_channel'] is not None:
                config['plot_mosaic']['origin_image'] = img_data[..., config['display_origin_image_channel']]

            config['plot_mosaic']['dataset'] = dataset
            config['plot_mosaic']['name_ID'] = name_ID

            if isinstance(config['select_image']['plot_mosaic'],str):
                plot_mosaic(config, dict_data[config['select_image']['plot_mosaic']],
                            image_type=config['select_image']['plot_mosaic'],
                            client_save_rootdir=client_save_rootdir, **config['plot_mosaic'])
            else:
                for name in config['select_image']['plot_mosaic']:
                    plot_mosaic(config, dict_data[name],
                                image_type=name,
                                client_save_rootdir=client_save_rootdir, **config['plot_mosaic'])

        elif figure_name == 'plot_by_slice':
            if config['colormap'] is not None:
                config['plot_by_slice']['colormap'] = (np.array(config['colormap']) * 255).astype('int32')
            if config['display_origin_image_channel'] is not None:
                config['plot_by_slice']['origin_image'] = img_data[..., config['display_origin_image_channel']]
            config['plot_by_slice']['dataset'] = dataset
            config['plot_by_slice']['name_ID'] = name_ID


            input_img_list = [dict_data[select_img] for select_img in config['select_image']['plot_by_slice']]

            for img in input_img_list:
                if None is img:
                    raise ValueError('Nonetype can not be appeared in input_img_list! ')
            if not isinstance(config['plot_by_slice']['slice_'], int):  # a list of slices
                slices = config['plot_by_slice']['slice_']
                for slice in slices:
                    config['plot_by_slice']['slice_'] = slice
                    plot_by_slice(config, input_img_list, client_save_rootdir=client_save_rootdir,
                                          **config['plot_by_slice'])
            else:
                plot_by_slice(config, input_img_list, client_save_rootdir=client_save_rootdir,
                                  **config['plot_by_slice'])
        else:
            # add figure config here if more figure methods are added.
            pass
    return


def plot_figures_dataset(config, dict_data, dataset='0', client_save_rootdir=None):
    """
       Plot figure configuration of a dataset.
       :param config: type dict: config parameter
       :param dict_data: type dict of list of ndarray: plot data
       :param dataset: type str: name of the dataset
       :param client_save_rootdir: type str: root dir of saving plot files. None if use default directory
                                   which is defined in the yaml.
       :return:
       """

    if config['plot_figure'] == [] or config['plot_figure'] == {} or config['plot_figure'] is None:
        return
    elif isinstance(config['plot_figure'], str):
        config['plot_figure'] = [config['plot_figure']]
    for figure_name in config['plot_figure']:
        if figure_name == 'plot_area_ratio':
            input_img_list = [dict_data[select_img] for select_img in config['select_image']['plot_area_ratio']]

            # Check in Nonetype in input_img_list
            # (Mostly occurs when only predict img data is available, but label item appears in plot settings
            #  in the config.yaml)
            for img in input_img_list:
                if isinstance( img , list):
                    for  im in img:
                        if im is None:
                            raise ValueError('Nonetype can not be appeared in input_img_list! ')
                if img==None:
                    raise ValueError('Nonetype can not be appeared in input_img_list! ')
            config['plot_area_ratio']['dataset'] = dataset

            plot_area_ratio(config, input_img_list, client_save_rootdir=client_save_rootdir,
                            **config['plot_area_ratio'])
        else:
            # add figure config here if more figure methods are added.
            pass
    return
