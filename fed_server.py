import os

import argparse
from tabnanny import verbose
from typing import Dict, Optional, Tuple
from pathlib import Path


import tensorflow as tf
import flwr as fl

from fed_models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 

def main() -> None:

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--num_clients", type=int, required=True) # per scrivere la cartella giusta, temporaneo

    args = parser.parse_args()
    address = args.address

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    model = get_model() # same as fed_client.py


    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1, # era 0.3
        fraction_eval=0.0, # 0: disabled
        min_fit_clients=args.num_clients,
        #min_eval_clients=2,
        min_available_clients=args.num_clients,
        #eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:"+address,
        config={"num_rounds": 10},
        strategy=strategy,
        #certificates=(
            #Path(".cache/certificates/ca.crt").read_bytes(),
            #Path(".cache/certificates/server.pem").read_bytes(),
            #Path(".cache/certificates/server.key").read_bytes(),
        #),
    )


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    print("---------------------------------------------------------------------------------------------------------------")

    # Use the last 5k training examples as a validation set
    x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights,) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        #"local_epochs": 5
        "local_epochs": 1 if rnd < 2 else 10 # era else 2
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
