import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pickle

def get_csv_data(path, name):

    df = pd.read_csv(path + name)
    return df

def read_test_ids(config, dataset):
    ids_test =[]
    with open('.'+config['dir_dataset_info'] + '/split_paths_' + dataset + '.pickle', 'rb') as fp:
        split_path = pickle.load(fp)
        dataset_image_path = split_path['path_test_img']
        
    ## extract the ids
    for path in dataset_image_path:
    
        id = path[0].split('/')[5]
        ids_test.append(id)

    return ids_test

def plot_histogram(config, dataset, df, item_info, png_dir):
    hist1 = df['dice_lesion'].round(decimals=2).hist()
    plt.xlabel('dice score')
    plt.title('dice score histogram')
    #plt.set_xlim(3, 18)
    plt.ylim(0, 130)
    png_name = 'plot_' + dataset + '_' + item_info + '.png'
    plt.savefig(png_dir + png_name)
    plt.show()

def plot_tumor_volume_dice_score(config, dataset, df, item_info, png_dir):
    
    #y_axis = [np.log(df['num_lesions_manual']), np.log(df['num_lesions_prediction'])]
    #y_label = ['manual lesions', 'predicted lesions']
    #for y, label in zip(y_axis, y_label):
    plt.scatter(df['dice_lesion'].round(decimals=2), np.log(df['num_lesions_manual']))#,label=label)
        
    #plt.legend(loc='upper left')
    plt.xlabel('dice coefficient')
    plt.ylabel('log tumor volume')
    plt.title('Dice Coefficient vs Number of lesions')
    #plt.xlim(4, 15)
    #plt.ylim(4, 15)
    png_name = 'plot_' + dataset + '_' + item_info + '.png'
    plt.savefig(png_dir + png_name)
    plt.show()

def plot_combine_metabolic_tumor_volume(config, dataset, df1, df2, item_info, png_dir):
    fig = plt.figure()
    ax1= fig.add_subplot()
    ax1.scatter(np.log(df1['volume_manual']), np.log(df1['volume_prediction']), c='b', label='he_norm')
    ax1.scatter(np.log(df2['volume_manual']), np.log(df2['volume_prediction']), c='r', label='he_uni')
    plt.xlabel('log tumor volume manual')
    plt.ylabel('log tumor volume prediction')
    plt.title('Metabolic Tumor Volume')
    plt.xlim(4, 16)
    plt.ylim(4, 16)
    plt.legend(loc='upper left');
    png_name = 'plot_combine_' + dataset + '_' + item_info + '.png'
    plt.savefig(png_dir + png_name)
    plt.show()

def plot_metabolic_tumor_volume(config, dataset, df, item_info, png_dir):
    scatter = plt.scatter(np.log(df['volume_manual']), np.log(df['volume_prediction']))
    plt.xlabel('log tumor volume manual')
    plt.ylabel('log tumor volume prediction')
    plt.title('Metabolic Tumor Volume')
    plt.xlim(4, 16)
    plt.ylim(4, 16)
    png_name = 'plot_' + dataset + '_' + item_info + '.png'
    plt.savefig(png_dir + png_name)
    plt.show()

def plot_TLG(config, dataset, df, item_info, png_dir):
    #print(df['uptake_mean_manual'].dtypes)
    #print(df.dtypes)
    #print(df['uptake_mean_manual'])
    i=0
    for elem in df['uptake_mean_prediction']:
        #if elem == "--":
        #    print("element has to be changed")
        #    elem=0
        print("elem " + str(i) + " : ", elem )
        print("type of elem: ", type(elem))
        i=i+1
    df.loc[(df.uptake_mean_manual == '--'), 'uptake_mean_manual'] = 0.0001
    #df.loc[(df.volume_manual == '--'), 'volume_manual'] = 0.0001
    df.loc[(df.uptake_mean_prediction == '--'), 'uptake_mean_prediction'] = 0.0001
    #df.loc[(df.volume_prediction == '--'), 'volume_prediction'] = 0.0001
    i = 0
    for elem in df['uptake_mean_prediction']:
        # if elem == "--":
        #    print("element has to be changed")
        #    elem=0
        print("elem " + str(i) + " : ", elem)
        print("type of elem: ", type(elem))
        i = i + 1
    df.uptake_mean_manual = df.uptake_mean_manual.astype(float)
    df.uptake_mean_prediction = df.uptake_mean_prediction.astype(float)
    tlg_per_patient = df['volume_manual'].mul(df['uptake_mean_manual'].values)
    tlg_estimated_per_patient = df['volume_prediction'].mul(df['uptake_mean_prediction'].values)
    scatter = plt.scatter(np.log(tlg_per_patient), np.log(tlg_estimated_per_patient))
    plt.xlabel('log TLG per patient')
    plt.ylabel('log Estimated TLG per patient')
    plt.title('Total Tumor Glycolysis')
    plt.xlim(4, 17)
    plt.ylim(4, 17)
    png_name = 'plot_' + dataset + '_' + item_info + '.png'
    plt.savefig(png_dir + png_name)
    plt.show()

def get_config(config_path):
    with open(config_path, "r") as yaml_file:
        config = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    return config

def plot_validation(config, dataset):

    ids_test= read_test_ids(config, dataset)

    #path_csv_test = '.' + config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
    #               '/validation_result/csv_data/' + dataset + '/'

    path_csv_all = '.' + config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
                   '/validation_result/csv_data_all/' + dataset + '/'

    csv_name = 'validation.csv'

    png_dir = '.' + config['result_rootdir'] + '/' + config['exp_name'] + '/' + config['model'] + \
              '/figures/plot_validation_results/'

    if not os.path.exists(png_dir): os.makedirs(png_dir)

    #pet_csv = get_csv_data(path_csv_test, csv_name)
    pet_csv_all = get_csv_data(path_csv_all, csv_name)

    #df_0_79 = pet_csv_all[pet_csv_all['key'] == "5d9b602590"]

    pet_csv_test = pet_csv_all[pet_csv_all['key'].isin(ids_test)]

    #df_0_79_test = pet_csv_test[pet_csv_test['key'] == "5d9b602590"]
    
    #plot_histogram(config, dataset, pet_csv, "dice_score_histogram", png_dir)
    plot_histogram(config, dataset, pet_csv_test, "dice_score_histogram", png_dir)
    plot_histogram(config, dataset, pet_csv_all, "dice_score_histogram_all", png_dir)

    #plot_metabolic_tumor_volume(config, dataset, pet_csv,
    #                            "Metabolic_Tumor_Volume", png_dir)
    plot_metabolic_tumor_volume(config, dataset, pet_csv_test,
                                "Metabolic_Tumor_Volume", png_dir)
    plot_metabolic_tumor_volume(config, dataset, pet_csv_all,
                                "Metabolic_Tumor_Volume_All", png_dir)

    pet_csv_he_norm_dir = '.' + config['result_rootdir'] + '/' + "Melanoma_4" + '/' + config['model'] + \
                   '/validation_result/csv_data_all/' + dataset + '/'
    pet_csv_he_uni_dir = '.' + config['result_rootdir'] + '/' + "Melanoma_6" + '/' + config['model'] + \
                   '/validation_result/csv_data_all/' + dataset + '/'

    pet_csv_he_norm = get_csv_data(pet_csv_he_norm_dir, csv_name)
    pet_csv_he_uni = get_csv_data(pet_csv_he_uni_dir, csv_name)
    plot_combine_metabolic_tumor_volume(config, dataset, pet_csv_he_norm, pet_csv_he_uni,
                                "Metabolic_Tumor_Volume", png_dir)

    #plot_TLG(config, dataset, pet_csv,
    #         "TLG_test", png_dir)
    plot_TLG(config, dataset, pet_csv_test,
             "TLG_test", png_dir)
    plot_TLG(config, dataset, pet_csv_all,
                                "TLG_all", png_dir)

    #plot_tumor_volume_dice_score(config, dataset, pet_csv_test,
    #         "dice_coef_vs_tumor_vol", png_dir)
    #plot_tumor_volume_dice_score(config, dataset, pet_csv_all,
    #         "dice_coef_vs_tumor_vol", png_dir)

    print("validation histogram and metabolic tumor volume plots completed created and saved")


path_config = '../config/config_melanoma.yaml'
config = get_config(path_config)
plot_validation(config, dataset='MELANOM')