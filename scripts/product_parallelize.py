"""
This script takes a generic script and parallelizes by taking the cartesian product of experimental parameter choices.
This ideal in the case of a combinatorial experiment, e.g.,

param1 = iter(['a', 'b', 'c'])
param2 = iter([1, 2])

exp1: (param1='a', param2=1), exp2:  (param1='a', param2=2), exp3: (param1='b', param2=1), etc...

In order to paralellize over a parameter the parameter must be an iterator, i.e., it has a __next__ method, this is not
to be confused with an Iterable which has the potential to become an iterator.

An example of running product_parallelize.py is as follows:

python product_parallelize.py classify.py --exp_params /hdd/DataSci/test_data/Iris/Experiments/setosa_versicolor_classify_species_svm/setosa_versicolor_classify_species_svm_params.py

The results will be saved, per the "save" function provided in your script, for each combination of parameters. The standard
save name will be appended with an integer indicating the parallel run, e.g., save_name_0.blah, save_name_1.blah, etc...

A lookup table with the parameter values is saved as a .csv in the experiments results directory,
row i corresponds to the save_name_i.blah.
"""

# imports
import itertools
import os
import importlib
import ray
from inspect import signature
import sys
from tqdm import tqdm
from copy import deepcopy
from datasci.core.helper import batch_jobs_
from datasci.core.helper import module_from_path
import csv

# functions
def generate_iterable_for_ray(script):

    # create an iterable for the run parameters in the script
    options = []
    options_no_remote = []
    for param in signature(script.run).parameters.keys():
        # get param value
        value = eval("script." + param)

        # check if the parameter is an iterator
        if not hasattr(value, "__next__"):
            # cast the value as a singleton list and put into the object store
            value_remote = [ray.put(value)]
            value_no_remote = [value]
        else:
            value_remote = value_no_remote = value

        options.append(value_remote)
        options_no_remote.append(value_no_remote)

    # compute combinations
    combos = itertools.product(*options)
    combos_no_remote = itertools.product(*options_no_remote)

    return [i for i in combos], [i for i in combos_no_remote],


def compute_measure_futures(futures):
    print("Running parallel scripts...")
    bar = tqdm(total=len(futures))
    was_ready = []
    total = 0
    # get futures
    while True:
        ready, not_ready = ray.wait(futures)
        delta_ready = list(set(ready).difference(set(was_ready)))
        if len(delta_ready) > 0:
            bar.update(len(delta_ready))
            total = total + len(delta_ready)
        if total == bar.total:
            break
        was_ready = ready
    bar.close()

    return


# main body
if __name__ == "__main__":

    # trick the python script to use the correct arguments
    sys.argv = sys.argv[1:]

    # grab the script with its parameters
    script = module_from_path("script", sys.argv[0])
    script_path = os.path.abspath(script.__file__)
    script_name = script_path.split('/')[-1].split('.')[0]
    save = script.save
    run_params = signature(script.run).parameters.keys()

    # ray initiate ray
    ray.init(#local_mode=True
             )

    # generate all possible combination of arguments for script
    combos, combos_no_remote = generate_iterable_for_ray(script)

    # wrap the run method into a remote ray func
    @ray.remote
    def run(argv, script_path, *args: list):
        # mimics command line args
        sys.argv = argv

        # extract the script name and directory
        script_name = script_path.split('/')[-1].split('.')[0]
        script_dir = os.path.dirname(script_path)

        # add the directory to the remote system paths
        sys.path.append(script_dir)

        # grab the run method from the script
        f = getattr(importlib.import_module('.'.join(['scripts', script_name])), "run")

        return f(*args)

    # run the remote processes
    run_args = [[deepcopy(sys.argv), script_path, *option] for option in combos]
    futures = batch_jobs_(run,
                          run_args,
                          )

    # compute the options
    #compute_measure_futures(futures)

    # store the results
    results = ray.get(futures)

    # shutdown ray
    ray.shutdown()

    # save the results
    for i, result in enumerate(results):
        script.save(result, i)

    with open(os.path.join(script.results_dir, '_'.join([script.exp_name,
                                                         script_name,
                                                         'product_parallelize_args.csv']
                                                        )), "w") as f:
        writer = csv.writer(f)
        writer.writerows([run_params] + [[str(j) for j in i] for i in combos_no_remote])
















