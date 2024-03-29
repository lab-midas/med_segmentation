import tensorflow as tf
from med_io.pipeline import *
from models.ModelSet import *
from sklearn.model_selection import train_test_split
from util import *
from utils.TensorBoardTool import *
from plot.plot_figure import *
from tensorflow.keras.models import load_model
from med_io.keras_data_generator import DataGenerator, tf_records_as_hdf5, \
    save_used_patches_ids
from med_io.active_learning import CustomActiveLearner, query_selection, \
    choose_random_elements, query_random
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

    print("database: ", config['dataset'])
    print("type database: ", type(config['dataset']))
    for pickle_path, pickle_max_shape, dataset in zip(config['filename_tfrec_pickle'],
                                                      config['filename_max_shape_pickle'],
                                                      config['dataset']):
        if restore:
            # Resume dataset.
            with open(config['dir_model_checkpoint'] + '/' + config['exp_name'] + '/' + 'training_info.pickle',
                      'rb') as fp:
                restore_dataset = pickle.load(fp)['dataset']

                while True:
                    if restore_dataset != dataset:
                        command = input(
                            'Warning! The stored resuming dataset name last time is not coincident with the dataset this time,'
                            ' do you want to overwrite? (y/n)')
                        if command == 'y':
                            break
                        elif command == 'n':
                            dataset = restore_dataset
                            break
                        else:
                            print('Invalid command.')
                    else:
                        print('Resume training dataset: ', dataset, '...')
                        break

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
        if not os.path.exists(config['dir_model_checkpoint'] + '/' + config['exp_name']): os.makedirs(
            config['dir_model_checkpoint'] + '/' + config['exp_name'])
        checkpoint_path = config['dir_model_checkpoint'] + '/' + config['exp_name'] + '/cp_' + dataset + '_' + config[
            'model'] + '.hdf5'

        tb_tool = TensorBoardTool(config['dir_model_checkpoint'] + '/' + config['exp_name'])  # start the Tensorboard

        # Create a callback that saves the model's weights every X epochs.
        cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=False,
                                                          period=config['save_training_model_period']),
                       tf.keras.callbacks.TensorBoard(os.path.dirname(checkpoint_path), histogram_freq=1)]

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
            """
            The program on the end of each epoch,
            Add here if any progress are processed on the end of each epoch
            """
            def on_epoch_end(self, epoch, logs={}):

                if epoch % config['save_training_model_period'] == 0 and epoch != 0:
                    with open(config['dir_model_checkpoint'] + '/' + config['exp_name'] + '/training_info.pickle',
                              'wb') as fp:
                        pickle.dump({'epoch': epoch, 'dataset': dataset, 'model': config['model']}, fp,
                                    protocol=pickle.HIGHEST_PROTOCOL)
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

    """Internal function
        Train process"""

    # if active learning is configured use al training process otherwise normal
    if config['active_learning']:
        print('Using active learning loop for training')
        return train_al_process(config, model, paths_train_img, paths_train_label,
                                paths_val_img, paths_val_label, dataset, cp_callback, saver1)
    else:

        if dataset == 'MELANOM':
            print("reading pipeline for Melanom dataset")

            ds_train = pipeline_melanom(config, paths_train_img, paths_train_label,
                                        dataset=dataset, augment=True, training=True)

            ds_validation = pipeline_melanom(config, paths_val_img, paths_val_label,
                                             dataset=dataset, augment=False, training=True)

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
    """
    Train with Active Learning (AL): alternative to the train_process() function. Uses the same parameters.
    """
    # convert the tf_records data to hdf5 if this hasn't already happened
    print('Making shure data is available as hdf5 file')
    hdf5_path, train_ids, val_ids = tf_records_as_hdf5(paths_train_img, paths_train_label,
                                                       paths_val_img, paths_val_label,
                                                       config, dataset=dataset)

    # Define validation data DataGenerator (Sequence object)
    val_data = DataGenerator(hdf5_path, val_ids,
                             batch_size=config['evaluate_batch_size'],
                             dim=config['patch_size'],
                             n_channels=len(config['input_channel'][dataset]),
                             n_classes=len(config['output_channel'][dataset]),
                             steps_per_epoch=config['val_steps_per_epoch'])

    # choose patches from training data for initial training
    train_ids, init_ids = choose_random_elements(train_ids,
                                                 config['al_num_init_patches'])

    # save info of IDs and patches
    save_used_patches_ids(config, ['hdf5_file', 'train_ids', 'init_ids', 'val_ids'],
                          [(config['al_patches_data_dir'] + '/' + config['al_patches_data_file']),
                           train_ids, init_ids, val_ids],
                          first_time=True)

    # check if enough train patches are available
    assert len(train_ids) > config['al_iterations'] * config['al_num_instances'], \
        ('not enough training patches for these AL parameters! Reduce num of '
         'al iterations and/or num of instances queried every iteration.')

    # define arguments for fit in active learner
    fit_kwargs = {'callbacks': cp_callback + [saver1],
                  'shuffle': False,
                  'validation_data': val_data,
                  'validation_freq': config['validation_freq'],
                  'verbose': config['train_verbose_mode'],
                  'workers': config['al_num_workers'],
                  'use_multiprocessing': config['al_num_workers'] is not None}
    # Note: the epoch parameters are defined in the active learner

    # choose query strategy
    query_strategies = {'uncertainty_sampling': query_selection,
                        'random_sampling': query_random}
    query_strategy = query_strategies[config['query_strategy']]

    # instantiate an active learner that manages active learning
    print('Initializing active learner object, with {0} patches in pool'.format(len(train_ids)))
    learner = CustomActiveLearner(config, model, query_strategy, hdf5_path,
                                  train_ids, dataset, config['batch'],
                                  config['predict_batch_size'],
                                  init_ids=init_ids,
                                  train_steps_per_epoch=config['train_steps_per_epoch'],
                                  **fit_kwargs)

    for al_epoch in range(config['al_iterations']):
        print('AL epoch ' + str(al_epoch))

        # query new patches
        query_ids = learner.query(config=config,
                                  n_instances=config['al_num_instances'],
                                  al_epoch=al_epoch)

        # labeling of unlabeled data can later be implemented here

        # teach model with new patches and log the data
        learner.teach(query_ids, only_new=config['al_only_new'], **fit_kwargs)

    history = learner.histories


    return model, history


def train_config_setting(config, dataset):
    """
    Configuring parameter for training process
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

    print("max shape image: ", config['max_shape']['image'])
    print("max shape label: ", config['max_shape']['label'])
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
        # some pretrained models had already added background output.
        if config['model_add_background_output']:  # and (not config['train_premodel_add_background_output']):
            config['channel_label_num'] += 1

    print('channel_img,', config['channel_img_num'], 'channel_label,', config['channel_label_num'])
    return config


def k_fold_train_process(config, model, k_fold, paths, dataset, cp_callback, init_epoch, saver1):
    """
    K-fold training process

    :param config: type dict: config parameter
    :param model:  type tf.keras.Model, training model
    :param paths:  type dict of str: tfrecords path loaded from pickle file.
    :param dataset: type str: name of dataset
    :param cp_callback: type tf.keras.callbacks.ModelCheckpoint, training check point
    :param init_epoch:  type int, initial epoch.
    :return: models:  type tf.keras.Model, trained model
    :return: history type  list of float, metrics evaluating value from each epoch.
    """
    # history = None
    list_1 = list(zip(paths['path_train_val_img'], paths['path_train_val_label']))
    random.shuffle(list_1)
    divided_datapath = len(list_1) // k_fold
    assert (divided_datapath > 0)
    history = []
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

            model, history = train_process(config, model, paths_train_img, paths_train_label, paths_val_img,
                                           paths_val_label, dataset, cp_callback,
                                           saver1, k_fold_index=k,
                                           init_epoch=k * config['epochs'] + init_epoch)

        else:
            # establish one new model at each fold.
            model, hist = train_process(config, model, paths_train_img, paths_train_label, paths_val_img,
                                        paths_val_label, dataset, cp_callback,
                                        saver1, k_fold_index=k,
                                        init_epoch=init_epoch)
            history.append(hist)

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
