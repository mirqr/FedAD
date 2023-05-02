import argparse
import os
import time
from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score

import tensorflow as tf
import flwr as fl
import pandas as pd
import numpy as np
from sorcery import dict_of

from fed_models import *
from datautil import *


import runconfig

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # deactivate gpu 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, x_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception(
            "Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"] # vedi la strategia su serverf.fit_config.

        # Train the model using hyperparameters from config
        # Adapted to Autoencoder training (no labels)
        history = self.model.fit(
                self.x_train, 
                self.x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_split=0.1,
                #validation_data=(x_test, x_test)
                verbose = 0,
                #callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
                )

        
        #history = self.model.fit(self.x_train,            self.y_train,            batch_size,            epochs,            validation_split=0.1,        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            #"accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            #"val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, = self.model.evaluate(self.x_test, self.x_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test



def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--num_clients", type=str, required=True) # cartella con numero client ideale, temporaneo
    parser.add_argument("--out_fold", type=str, required=True) # 
    args = parser.parse_args()

    # Load and compile Keras model (same used in the server)
    model = get_model() 

    # Load a subset of CIFAR-10 to simulate the local data partition
    x_train, x_test, y_train, y_test = load_partition(args.partition, args.data_name) 
    # nel nostro caso Y_OUT. Le y non le usiamo in fed, le usiamo noi in ogni singolo thread"""

    # Start Flower client
    client = CifarClient(model, x_train, x_test)

    address = args.address
    fl.client.start_numpy_client(
        server_address="localhost:"+address,
        client=client,
       # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )
    
    # compute evaluation and save results
    path = 'out'+args.num_clients+'/'+'output_'+args.out_fold+'/'+args.data_name+'/'
    res_and_save(model,x_train,x_test,y_train, y_test, args.partition, path)

    # save model if you want 
    # model.save('saved_model_local/'+args.data_name+'/'+args.partition)
    

    # Here Federeated Learning has finished
    # Now perform only local training and evaluation with the same model used in FL


    local_training = True
    if local_training:
        print('Start local training', args.partition, args.data_name)
        model_local = get_model() # same used in the  as FL
        history = model_local.fit(
                    x_train, 
                    x_train,
                    epochs=40,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.1,
                    #validation_data=(x_test, x_test)
                    verbose =0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
                    )

        # compute evaluation and save results
        path = 'out'+args.num_clients+'/'+'output_local/'+args.data_name+'/'
        res_and_save(model_local,x_train,x_test,y_train, y_test, args.partition, path)

        # save model if you want 
        # model_local.save('saved_model_local/'+args.data_name+'/'+args.partition)




def res_and_save(model,x_train,x_test,y_train, y_test, partition, path):
    res = my_predict(model,x_train,x_test,y_train, y_test)
    res['dataset'] = partition
    df_res = pd.DataFrame.from_dict([res]).round(4)

    os.makedirs(path, exist_ok=True)
    df_res.to_csv(path+partition+'.csv')


def distancess(x,x_pred):
    euclidean_sq = np.square(x - x_pred)
    distances = np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()
    return distances

def my_predict(model, x_train, x_test, y_train, y_test):
    x_train_pred = model.predict(x_train)
    distances = distancess(x_train, x_train_pred)
    contamination = runconfig.get_config_dict()["outlier_fraction"]
    threshold = np.percentile(distances, 100 * (1 - contamination))

    x_test_pred = model.predict(x_test)

    distances = distancess(x_test, x_test_pred)
    labels = (distances > threshold).astype('int').ravel()



    y_true = y_test
    y_score = distances
    y_pred = labels
    aucroc = roc_auc_score(y_true, y_score)
    #aucpr = round(average_precision_score(y_true, y_score, pos_label=1)
    f1in = f1_score(y_true, y_pred, pos_label=0)
    f1out = f1_score(y_true, y_pred, pos_label=1)
    acc = accuracy_score(y_true, y_pred, normalize=True)
    return dict_of(aucroc, f1in, f1out, acc)




    



def load_partition(idx: str, data_name:str = 'mnist'):
    print("LOAD Partition", idx, data_name)
    
    x_train, y_train, x_test, y_test = get_dataset(data_name, flatten_and_normalize=True)
    n_features = np.prod(x_train.shape[1:])   

    datasets = partition(x_train, y_train, num_clients_per_class=runconfig.get_config_dict()['num_clients_per_class'])
    x_train_local = datasets[idx].iloc[:,:n_features].to_numpy()
    y_train_out_local =  datasets[idx]['y_out'].to_numpy()
    #print(idx,'train_set\n',datasets[idx]['y_class'].value_counts())

    inlier_class = datasets[idx]['y_class'].mode()[0]  # inlier class = most common class
    
    
    # partition the test set like the train set
    #datasets_test = partition(x_test, y_test, runconfig.get_config_dict()['num_clients_per_class'])        
    #x_test_local = datasets_test[idx].iloc[:,:n_features].to_numpy()
    #y_test_out_local =  datasets_test[idx]['y_out'].to_numpy()
    #print(idx,'test_set\n',datasets_test[idx]['y_class'].value_counts())

    # instead of partitioning the test set like the train set, we take more samples from the inlier class
    datasets_test = partition(x_test, y_test, num_clients_per_class=1)        
    idx_test = str(inlier_class)+'_0'
    x_test_local = datasets_test[idx_test].iloc[:,:n_features].to_numpy()
    y_test_out_local =  datasets_test[idx_test]['y_out'].to_numpy()

    
    
    
    


    #return x_train_local, x_test_local
    return x_train_local, x_test_local, y_train_out_local, y_test_out_local





if __name__ == "__main__":
    main()
    
