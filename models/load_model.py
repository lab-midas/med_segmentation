from models import Premodel_Custom_Class
from models.Premodel_Custom_Class import *
from models.ModelSet import *
from models.Premodel_Set import *
from tensorflow.keras.models import load_model
import tensorflow as tf


def load_model_file(config, dataset, compile=False):
    if config['load_weights_only'] == True:
        if 'premodel' in config['model']:

            # Load the model weights, the model is defined in Premodel_Set.py
            call_model = getattr(Premodel_Set, config['model'])
            model, _ = call_model(self=Premodel_Set, config=config)
            print('Now dealing dataset ', dataset, 'from the model ', config['model'], '...')
            print(config['saved_premodels_path'])
            model.load_weights(config['saved_premodels_path'])

        else:
            # Load the model weights, the model is defined in ModelSet.py
            call_model = getattr(ModelSet, config['model'])
            model, _ = call_model(self=ModelSet, config=config)
            print('Now dealing dataset ', dataset, 'from the model ', config['model'], '...')
            print('The model path: ', config['saved_models_dir'] +'/'+config['exp_name']+ '/' + config['model'] + '/' + dataset + '.h5')
            model.load_weights(config['saved_models_dir'] +'/'+ config['exp_name']+'/' + config['model'] + '/' + dataset + '.h5')
    else:
        # Load the model without knowing its structure.
        # custom_object is not implement
        if config['model'] is None:
            config['model'] = 'custom_model'
        if config['custom_layer'] is None:
            model = load_model(config['saved_premodels_path'], compile=False)

        else:
            if isinstance(config['custom_layer'], str): config['custom_layer'] = [config['custom_layer']]
            custom_object = dict()
            for obj in config['custom_layer']:
                layer = getattr(Premodel_Custom_Class, obj)
                custom_object[obj] = layer

            model = load_model(config['saved_premodels_path'], custom_objects=custom_object, compile=False)
        if compile:

            in_ = model.get_layer(name='input_X')
            out = model.get_layer(name='output_Y')
            if config['feed_pos']:
                in_pos = model.get_layer(name='input_position')
                model, _ = create_and_compile_model([in_, in_pos], out, config, premodel=model)
            else:
                model, _ = create_and_compile_model(in_, out, config, premodel=model)

    return model
