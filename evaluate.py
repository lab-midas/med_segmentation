import tensorflow as tf
from med_io.pipeline import *
from models.ModelSet import *
from models.loss_function import *
from models.metrics import *
from sklearn.model_selection import train_test_split
from util import *
from tensorflow.keras.models import load_model
import pickle
import os
from models.load_model import load_model_file
from predict import channel_config

from med_io.pipeline_melanom import *


def evaluate(config, datasets=None):
    """
    Evaluate the test data by given saved model.
    :param config: type dict: config parameter
    :param datasets: type list of str: list of dataset names
    :return: lists_loss_and_metrics: type list: evaluate result (losses and metrics)
    """
    lists_loss_and_metrics=[]
    for dataset in datasets:
        #  Get test dataset_image_paths and dataset_label_path
        if not os.path.exists(config['dir_dataset_info']+'/split_paths_'+dataset+'.pickle'):
            raise FileNotFoundError('Paths of dataset   `config[dir_dataset_info]/split_paths.pickle`are not found! ')
        with open(config['dir_dataset_info']+'/split_paths_'+dataset+'.pickle', 'rb') as fp:
            split_path = pickle.load(fp)
            dataset_image_path = split_path['path_test_img']
            dataset_label_path = split_path['path_test_label']

        # Set the config files
        config = channel_config(config, dataset, evaluate=True)
        # create pipeline dataset
        ds_test = pipeline_melanom(config, dataset_image_path, dataset_label_path, dataset=dataset, evaluate=True)

        #ds_test = pipeline(config, dataset_image_path, dataset_label_path, dataset=dataset)
        # Choose the training model.
        model = load_model_file(config, dataset)

        ## get the names of metrics used
        list_metrics = model.metrics_names
        print(list_metrics)

        print('Now evaluating data ', dataset,' ...')

        # Fit training & validation data into the model
        #print(ds_test[0].shape)
        #print("Size of dataset: ", len(ds_test))
        list_loss_and_metrics = model.evaluate(ds_test,verbose=config['evaluate_verbose_mode'])
        lists_loss_and_metrics.append(list_loss_and_metrics)

        #print("printing metrics from evaluation.......")
        #print(list_loss_and_metrics)

        path_pickle = config['result_rootdir'] + '/' + config['exp_name']+ '/' + config['model']+'/evaluate_loss_and_metrics/'
        if not os.path.exists(path_pickle): os.makedirs(path_pickle)

        dictionary=dict()
        # Save loss
        #dictionary['evaluate_loss'] = list_loss_and_metrics[0]
        #dictionary['evaluate_acc'] = list_loss_and_metrics[1]
        #dictionary['evaluate_dicecoef'] = list_loss_and_metrics[2]
        # Save metrics
        for item, value in zip(list_metrics, list_loss_and_metrics):
            dictionary['evaluate_'+item] = value

        print(dictionary)

        with open(path_pickle+dataset + '.pickle', 'wb') as fp:
            pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print('Sucessfully save the evaluate loss and metrics of  ' + dataset + '.')

        print('Evaluating data ', dataset,'is finished.')

    return lists_loss_and_metrics
