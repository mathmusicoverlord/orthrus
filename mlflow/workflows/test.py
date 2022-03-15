"""Script for testing parallelism of MLFlow."""

# consts
PARAMS_0 = {'param_0_0':0, 'param_0_1':1, 'param_0_2':3, 'param_0_3':4}
PARAMS_1 = {'param_0_0':5, 'param_0_1':6, 'param_0_2':7, 'param_0_3':8}

# import
import mlflow
import time
from multiprocessing.pool import ThreadPool
import ray
from itertools import product

@ray.remote
def log_param(k, v):
    with mlflow.start_run(nested=True): 
        print(f"Logging param {k} with value {v}...")  
        mlflow.log_param(k, v) 

if __name__ == "__main__":

    # start mlflow
    results = []
    with mlflow.start_run():
        for (k0, v0), (k1, v1) in product(PARAMS_0.items(), PARAMS_1.items()):
            # log the params
            results.append(log_param.remote(k0, v0))
            
            results.append(log_param.remote(k1, v0 + v1))
    
    # wait on results
    ray.get(results)


