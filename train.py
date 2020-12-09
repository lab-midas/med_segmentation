import tensorflow as tf
from med_io.pipeline import *
from models.ModelSet import *
from sklearn.model_selection import train_test_split
from util import *
from plot.plot_figure import *
from tensorflow.keras.models import load_model
from med_io.keras_data_generator import DataGenerator, tf_records_as_hdf5
from med_io.active_learning import PatchPool, query_training_patches
from models.load_model import load_model_file
import pickle
import datetime
import os
import random


def train(config, restore=False):
    """
    Train the dataset from given paths of dataset.
    :param config: type dict: config parameter
    :param restore: type bool, True if resume training from the last checkpoint
    :return: models:  type list of model, trained model
    :return: histories type list of of list of float, metrics evaluating value from each epoch.
    """
    models, histories = [], []
    for pickle_path, pickle_max_shape, dataset in zip(config['filename_tfrec_pickle'],
                                                      config['filename_max_shape_pickle'],
                                                      config['dataset']):
        if restore:
            # Resume dataset.
            with open(config['dir_model_checkpoint'] + '/' + config['exp_name'] + '/' + 'training_info.pickle',
                      'rb') as fp:
                restore_dataset = pickle.load(fp)[dataset]
                if restore_dataset != dataset:
                    continue
                else:
                    print('Resume training dataset: ', dataset, '...')

        config = train_config_setting(config, dataset)

        # Load split (training, validation, test) tfrecord paths.
        if config['read_body_identification']:
            split_filename = config['dir_dataset_info'] + '/split_paths_' + dataset + '_bi.pickle'
        else:
            split_filename = config['dir_dataset_info'] + '/split_paths_' + dataset + '.pickle'
        with open(split_filename, 'rb') as fp:
            paths = pickle.load(fp)

        # Choose the training model.
        if not config['train_premodel']:
            call_model = getattr(ModelSet, config['model'])
            model, list_metric_names = call_model(self=ModelSet, config=config)
        else:  # load pre-trained model
            model = load_model_file(config, dataset, compile=True)

        print(model.summary())
        # Create checkpoint for saving model during training.
        if not os.path.exists(config['dir_model_checkpoint'] + '/' + config['exp_name']):
            os.makedirs(config['dir_model_checkpoint'] + '/' + config['exp_name'])
        checkpoint_path = config['dir_model_checkpoint'] + '/' + config['exp_name'] + '/cp.hdf5'

        # Create a callback that saves the model's weights every X epochs.
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=False,
                                                         period=config['save_training_model_period'])

        # Initial epoch of training data
        init_epoch = 0
        if restore:
            # Resume saved epoch.
            model.load_weights(checkpoint_path)
            with open(config['dir_model_checkpoint'] + '/' + config['exp_name'] + '/' + 'training_info.pickle',
                      'rb') as fp:
                init_epoch = pickle.load(fp)['epoch'] + 1
        restore = False

        # Log data at end of training epoch
        class Additional_Saver(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if epoch % config['save_training_model_period'] == 0 and epoch != 0:
                    with open(config['dir_model_checkpoint'] + '/' + config['exp_name'] + '/training_info.pickle',
                              'wb') as fp:
                        pickle.dump({'epoch': epoch, 'dataype': dataset}, fp, protocol=pickle.HIGHEST_PROTOCOL)
                if not os.path.exists('train_record'): os.makedirs('train_record')
                file1 = open('train_record/' + config['model'] + '_' + dataset + ".txt", "a+")
                now = datetime.datetime.now()
                file1.write('dataset: ' + dataset + ', Epoch: ' + str(epoch) +
                            ', Model: ' + config['model'] + ', time: ' + now.strftime("%Y-%m-%d %H:%M:%S") +
                            ', pid:' + str(os.getpid()))
                file1.write("\n")
                file1.close()

        saver1 = Additional_Saver()

        print('Now training data: ', dataset)
        k_fold = config['k_fold'][dataset]
        history_dataset = []
        if k_fold is not None:
            model, history = k_fold_train_process(config, model, k_fold, paths, dataset, cp_callback, init_epoch,
                                                  saver1)

        else:
            # without k-fold
            model, history = train_process(config, model, paths['path_train_img'], paths['path_train_label'],
                                           paths['path_val_img'], paths['path_val_label'],
                                           dataset, cp_callback, saver1,
                                           k_fold_index=0, init_epoch=init_epoch)
            history_dataset.append(history)
        saved_model_path = config['saved_models_dir'] + '/' + config['exp_name'] + '/' + config['model']
        if not os.path.exists(saved_model_path): os.makedirs(saved_model_path)
        # Save the model when training process is finished.
        model.save(saved_model_path + '/' + dataset + '.h5')
        print('Training data ', dataset, 'is finished')

        models.append(model)
        histories.append(history_dataset)
    return models, histories


def train_process(config, model, paths_train_img, paths_train_label, paths_val_img, paths_val_label, dataset,
                  cp_callback,
                  saver1, k_fold_index=0, init_epoch=0):
    """Internal function"""

    # if active learning is configured use al training process otherwise normal
    if config['active_learning']:
        return train_al_process(config, model, paths_train_img, paths_train_label,
                                paths_val_img, paths_val_label, dataset, cp_callback, saver1)
    else:
        ds_train = pipeline(config, paths_train_img, paths_train_label, dataset=dataset)

        ds_validation = pipeline(config, paths_val_img, paths_val_label, dataset=dataset)

        # Fit training & validation data into the model

        history = model.fit(ds_train,
                            epochs=config['epochs'] + init_epoch,
                            steps_per_epoch=config['train_steps_per_epoch'],
                            callbacks=[cp_callback, saver1],
                            initial_epoch=init_epoch,
                            validation_data=ds_validation,
                            validation_steps=config['val_steps_per_epoch'],
                            validation_freq=config['validation_freq'],
                            verbose=config['train_verbose_mode'])
        print(history.history)
        # Save the histories and plot figures
        save_histories_plot_images(history, config=config, dataset=dataset, mode='train_val', k_fold_index=k_fold_index)
        return model, history


def train_al_process(config, model, paths_train_img, paths_train_label, paths_val_img, paths_val_label, dataset,
                     cp_callback,
                     saver1, k_fold_index=0, init_epoch=0):

    # convert the tf_records data to hdf5 if this hasn't already happened
    test_path, test_trainid, test_valid = tf_records_as_hdf5(paths_train_img, paths_train_label, paths_val_img,
                                                             paths_val_label, config, dataset=dataset)
    # create the DataGenerator objects for the train process
    #test_generator = DataGenerator(test_path, test_trainid, n_channels=4, n_classes=4, batch_size=2)

    #learner = modAL.ActiveLearner(estimator=model, X_training=initial_data)


    model.fit = 'TBD'
    history = 'TBD'
    return model, history

def train_config_setting(config, dataset):
    """
    Configuring parameter for training
    :param config: type dict: config parameter
    :param dataset: type str: dataset  name
    :return: config: type dict: config parameter
    """
    # Load max shape & channels of images and labels.
    if config['read_body_identification']:
        filename_max_shape = config['dir_dataset_info'] + '/max_shape_' + dataset + '_bi.pickle'
    else:
        filename_max_shape = config['dir_dataset_info'] + '/max_shape_' + dataset + '.pickle'
    with open(filename_max_shape, 'rb') as fp:
        config['max_shape'] = pickle.load(fp)
    # Get the amount of input and output channel
    # config[channel_img]: channel amount of model input, config[channel_label]: channel amount of model output
    config['channel_img_num'], config['channel_label_num'] = config['max_shape']['image'][-1], \
                                                             config['max_shape']['label'][
                                                                 -1]
    if config['input_channel'][dataset] is not None:
        config['channel_img_num'] = len(config['input_channel'][dataset])

    if not config['read_body_identification']:
        if config['output_channel'][dataset] is not None:
            config['channel_label_num'] = len(config['output_channel'][dataset])

        # output channel+1 if the model output background channel (if the stored labels have no background channels)
        if config['model_add_background_output']:
            config['channel_label_num'] += 1

    print('channel_img,', config['channel_img_num'], 'channel_label,', config['channel_label_num'])
    return config


def k_fold_train_process(config, model, k_fold, paths, dataset, cp_callback, init_epoch, saver1):
    """
    K-fold training

    :param config: type dict: config parameter
    :param model:  type tf.keras.Model, training model
    :param paths:  type dict of str: tfrecords path loaded from pickle file.
    :param dataset: type str: name of dataset
    :param cp_callback: type tf.keras.callbacks.ModelCheckpoint, training check point
    :param init_epoch:  type int, initial epoch.
    :return: models:  type tf.keras.Model, trained model
    :return: history type  list of float, metrics evaluating value from each epoch.
    """
    history = None
    list_1 = list(zip(paths['path_train_val_img'], paths['path_train_val_label']))
    random.shuffle(list_1)
    divided_datapath = len(list_1) // k_fold
    assert (divided_datapath > 0)
    for k in range(k_fold):
        # Split train and eva
        list_val = list_1[k * divided_datapath:(k + 1) * divided_datapath]
        list_train = list_1[0:k * divided_datapath] + list_1[(k + 1) * divided_datapath:len(list_1)]

        print('k_fold', k, ' list_val:', list_val, ' list_train:', list_train)

        [paths_train_img, paths_train_label] = zip(*list_train)
        [paths_val_img, paths_val_label] = zip(*list_val)

        print('Now training data:', dataset, ', k fold: ', k, ' ...')
        if not config['k_fold_merge_model']:

            # train all k-fold on one model
            model, history_curr = train_process(config, model, paths_train_img, paths_train_label, paths_val_img,
                                                paths_val_label, dataset, cp_callback,
                                                saver1, k_fold_index=k,
                                                init_epoch=k * config['epochs'] + init_epoch)
            history.append(history_curr)

        else:
            # establish one new model at each fold.
            model, history_curr = train_process(config, model, paths_train_img, paths_train_label, paths_val_img,
                                                paths_val_label, dataset, cp_callback,
                                                saver1, k_fold_index=k,
                                                init_epoch=init_epoch)

            history.append(history_curr)
            # save model
            saved_model_path = config['saved_models_dir'] + '/' + config['exp_name'] + '/' + config['model']
            if not os.path.exists(saved_model_path): os.makedirs(saved_model_path)
            # Save the model when training process is finished.
            model.save(saved_model_path + '/' + dataset + 'k_fold_' + str(k) + '.h5')

            if k != k_fold - 1:
                # create a new model for next k-fold.
                if not config['train_premodel']:
                    call_model = getattr(ModelSet, config['model'])
                    model, list_metric_names = call_model(self=ModelSet, config=config)
                else:
                    model = load_model_file(config, dataset, compile=True)
    return model, history
