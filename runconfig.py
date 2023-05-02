######### Shared config #########
config_dict = {
    #### Data ####

    "seed": 42,  # set random seed
    "num_clients_per_class": 9,  # choices: Int
    "outlier_fraction": 0.1,  # choices: Float
    "association_threshold": 0.06,  
    "step1_path" : './data_step1_models/',
    "step2_path" : './data_step2_prediction/',
    "dataset_name": 'fashion_mnist',

}




def get_config_dict():
    return config_dict

