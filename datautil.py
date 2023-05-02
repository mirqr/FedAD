from math import floor
import os

import numpy as np
import pandas as pd
from joblib import dump, load
import tensorflow as tf

import runconfig


# "Device" class
# A class that represents a devices. It helps to manage the step 1 of the algorithm
class Dev:
    def __init__(self, name: str, dataset_name: str, x: pd.DataFrame, y_class):

        self.config = runconfig.get_config_dict()
        
        self.name = name
        self.dataset_name = dataset_name
        self.x = x.copy()
        self.y_class = y_class.copy()

        # inlier class = most common class
        # outlier class = all other classes
        inlier_class = y_class.mode()[0]  
        self.y_out = y_class.apply(lambda x: 0 if x == inlier_class else 1) 

    def getX_y_class(self):
        return pd.concat(self.x, self.y)

    def model_fit(self, model, num_clients_per_class=0, reset=False):
        clf_name = str(model).partition("(")[0]

        self.num_clients_per_class = num_clients_per_class
        
        root = self.config['step1_path']
        pth = "" if self.num_clients_per_class == 0 else str(self.num_clients_per_class)
        path = ( root + pth + "/" + self.dataset_name + "/" + clf_name + "/" )
        filename = str(self.name) + ".joblib"
        fullname = path + filename
        
        if not os.path.exists(fullname) or reset:
            #os.makedirs(os.path.dirname(fullname), exist_ok=True)
            os.makedirs(path, exist_ok=True) 
            print("Fitting and writing model", fullname)
            model.fit(self.x.to_numpy())
            dump(model, fullname)
        else:
            print("Reading model", fullname)
            model = load(fullname)
        self.model = model

    def write_other_models(self, devs: list, reset=False):
        clf_name = str(self.model).partition("(")[0]
        num_received = len(devs)  # used in path
        root = self.config["step2_path"]
        for sender in devs:  # device that sends its trained model
            r = self.name  # receiver
            s = sender.name
            path = (root + self.dataset_name + "/" + str(num_received) + "/" + clf_name + "/")
            filename = r + "-" + s + ".joblib"
            fullname = path + filename
            if not os.path.exists(fullname) or reset:
                y_pred = self.senders[sender]
                print("Writing", fullname)
                os.makedirs(path, exist_ok=True)
                dump(y_pred, fullname)


    def set_other_models(self, devs: list, reset=False, write=True):
        self.senders = {}
        clf_name = str(self.model).partition("(")[0]
        num_received = len(devs)  # mettiamo anche questo nel path
        root = self.config["step2_path"]

        for sender in devs:  # Device that sends its trained model
            r = self.name  # receiver
            s = sender.name
            pth = "" if self.num_clients_per_class == 0 else str( self.num_clients_per_class)
            path = ( root + pth + "/predictions/" + self.dataset_name + "/" + str(num_received) + "/" + clf_name + "/" )
            filename = r + "-" + s + ".joblib"
            fullname = path + filename
            if write:
                if not os.path.exists(fullname) or reset:
                    y_pred = sender.model.predict(self.x)
                    # print('Writing_comp', fullname)
                    os.makedirs(path, exist_ok=True)  
                    dump(y_pred, fullname)
                else:
                    try:
                        # print('Reading', fullname)
                        y_pred = load(fullname)
                    except EOFError as e: # if reading fails, predict again
                        print("ERROR", e)
                        y_pred = sender.model.predict(self.x)
                        # print('Writing', fullname)
                        os.makedirs(path, exist_ok=True)
                        dump(y_pred, fullname)
            else :
                y_pred = sender.model.predict(self.x)
            # careful
            self.senders[sender] = y_pred
            if r == s:
                self.y_pred = y_pred
        print(r, "finished to predict", num_received, "models")

    def get_senders_perc_normal(self):
        assert bool(self.senders)  # empty dict is false
        res_inlier = {}
        # res_outlier = {}
        for sender, y_pred in self.senders.items():
            x = y_pred
            res_inlier[sender] = np.count_nonzero(x == 0) / len(x)  #  inlier perc
            # res_outlier[sender] = np.count_nonzero(x == 1)/len(x) #  outlier perc
        return res_inlier

    def get_devs_federated(self, association_threshold=0.08):
        res_inlier = self.get_senders_perc_normal()
        # y_pred_local = self.model.predict(self.x)
        y_pred_local = self.y_pred

        local_inlier = np.count_nonzero(y_pred_local == 0) / len(y_pred_local)

        devs_federated = []
        for sender, y_pred in self.senders.items():
            x = y_pred
            sender_inlier = np.count_nonzero(x == 0) / len(x)
            # if in_range(sender_inlier, local_inlier, association_threshold):
            # devs_federated.append(sender)

            sender_local_pred = np.count_nonzero(sender.y_pred == 0) / len(x)  # y_pred locale del sender
            # my models, data of sender
            a = np.count_nonzero(sender.senders[self] == 0) / len(x)

            if in_range(sender_inlier, local_inlier, association_threshold) and in_range(a, sender_local_pred, association_threshold):
                devs_federated.append(sender)

        return devs_federated

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def in_range(x, ref, association_threshold):
    return ref - (ref * association_threshold) <= x <= ref + (ref * association_threshold)


# given a dataset x,y, partition it such that
# - for each class there are "num_clients_per_class clients" partition
# - each partition has (1.0 - outlier_fraction) inliers samples 
# - total number of partitions = num_clients_per_class * num_classes
def partition(x, y, num_clients_per_class=20, outlier_fraction=0.1,  random_state=runconfig.get_config_dict()["seed"]):
    df = pd.DataFrame(x)
    df["y_class"] = y
    class_list = list(np.unique(y))
    num_class = len(class_list)

    m1 = df.sample(frac = 1.0 - outlier_fraction, random_state=random_state)
    m2 = df.drop(m1.index).copy()
    #m1['y_class'].value_counts().plot.bar()
    #m2['y_class'].value_counts().plot.bar()
    inlier = {}
    
    datasets = {}
    for c in range(0, num_class):
        # inlier[0] list: num_clients_per_class (es 9) split della classe 0
        inlier[c] = np.array_split(m1.loc[m1["y_class"] == c], num_clients_per_class)

    for c in range(0, num_class):
        for j in range(0, num_clients_per_class):
            key = str(c) + "_" + str(j)
            #print(key)
            ii = inlier[c].pop(0)
            n_ii = len(ii)
            n_oo = floor(outlier_fraction * (n_ii / (1-outlier_fraction)))
            #print('leng',len(oo),len(ii))
            try:
                oo = m2.loc[m2["y_class"] != c].sample(n = n_oo, random_state=random_state)
            except ValueError:
                print("request, remaining",len(oo),len(m2.loc[m2["y_class"] != c]))
                oo = m2.loc[m2["y_class"] != c].sample(frac=1,random_state=random_state) # take all remaining
            m2=m2.drop(oo.index)
            len(m2)
            dd = pd.concat([ii, oo])
            dd = dd.sample(frac=1, random_state=random_state) #shuffle
            #len(out[c])
            dd["y_out"] = dd["y_class"].apply(lambda x: 0 if x == c else 1)
            datasets[key] = dd
    return datasets



def get_dataset(dataset_name: str, flatten_and_normalize=False):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset_name == 'mnist_and_fmnist_merged': #  20 classes
        (x_train0, y_train0), (x_test0, y_test0) = tf.keras.datasets.mnist.load_data()
        (x_train1, y_train1), (x_test1, y_test1) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.concatenate((x_train0, x_train1))
        y_train = np.concatenate((y_train0, y_train1+10)) # shift labels by 10
        x_test = np.concatenate((x_test0, x_test1))
        y_test = np.concatenate((y_test0, y_test1+10)) # shift labels by 10
    else:
        raise('Bad dataset name', dataset_name)
    
    # normalization hardcoded for mnist-like datasets
    if flatten_and_normalize:
        # flatten and normalize
        n_features = np.prod(x_train.shape[1:])
        x_train = x_train.reshape(x_train.shape[0], n_features) / 255.0
        x_test = x_test.reshape(x_test.shape[0], n_features) / 255.0

    return x_train, y_train, x_test, y_test