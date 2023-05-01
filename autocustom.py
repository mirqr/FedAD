from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf

import pyod

from pyod.utils.utility import check_parameter
from pyod.utils.stat_models import pairwise_distances_no_broadcast

from pyod.models.base import BaseDetector


from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2DTranspose, Conv2D, Reshape, Flatten, MaxPooling2D, UpSampling2D, MaxPool2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mean_squared_error


# custom autoencoder class inherited from BaseDetector class in pyod
# used to create a custom model to be used transparently in the step 1
class AutoEncoderCustom(BaseDetector):

    def __init__(self, hidden_neurons=None,
                 hidden_activation='relu', output_activation='sigmoid',
                 loss=mean_squared_error, optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1):
        super(AutoEncoderCustom, self).__init__(contamination=contamination)
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state

        # default values # TODO delete. Not needed here
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32, 32, 64]

        # Verify the network design is valid
        if not self.hidden_neurons == self.hidden_neurons[::-1]:
            print(self.hidden_neurons)
            raise ValueError("Hidden units should be symmetric")

        self.hidden_neurons_ = self.hidden_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    # Put your model here - Called by pyod 'fit' function 
    # hardcoded for mnist
    def _build_model(self):
        model = Sequential([
            Reshape((28, 28,1)),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same"),

            Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
            Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
            Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            Flatten()
        ])
        model.build(input_shape=(None,self.n_features_))
        model.compile(loss=self.loss, optimizer=self.optimizer)

        if self.verbose >= 1:
            print(model.summary())
        return model



  

    # noinspection PyUnresolvedReferences
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # Verify and construct the hidden units
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        # Shuffle the data for validation as Keras do not shuffling for
        # Validation Split
        np.random.shuffle(X_norm)

        # Validate and complete the number of hidden neurons
        if np.min(self.hidden_neurons) > self.n_features_:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")
        self.hidden_neurons_.insert(0, self.n_features_)

        # Calculate the dimension of the encoding layer & compression rate
        self.encoding_dim_ = np.median(self.hidden_neurons)
        self.compression_rate_ = self.n_features_ // self.encoding_dim_

        # Build AE model & fit with X
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X_norm, X_norm,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose,
                                        #callbacks=[EarlyStopping(monitor='loss', patience=3)]
                                        ).history #TODO aggiunto
        # Reverse the operation for consistency
        self.hidden_neurons_.pop(0)
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.predict(X_norm,verbose = self.verbose)
        self.decision_scores_ = pairwise_distances_no_broadcast(X_norm,
                                                                pred_scores)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        # Predict on X and return the reconstruction errors
        pred_scores = self.model_.predict(X_norm, verbose = self.verbose)
        return pairwise_distances_no_broadcast(X_norm, pred_scores)
