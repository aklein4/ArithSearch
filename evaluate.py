
import multivar_horner

from sparse_poly import SparsePoly
from simple_search import SimpleSearch

import csv
import numpy as np

SAVE_FOLDER = "./evaluation_data/"


def save_data(data: dict, filename: str):

    data_len = len(list(data.values())[0])
    for val in data.values():
        if len(val) != data_len:
            raise ValueError("data must be same length.")

    header = list(data.keys())

    # write to csv
    with open(SAVE_FOLDER+filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        
        # put header at top header
        spamwriter.writerow(header)

        for i in range(data_len):
            spamwriter.writerow([data[h][i] for h in header])


def command_wrap(command):
    outputs = {}

    for func in command.keys():

        if func == "horner":
            target = command[func]["target"]
            coefs = []
            keys = []
            for k in target.dict.keys():
                coefs.append(target[k])
                keys.append(k)
            horner = multivar_horner.HornerMultivarPolynomialOpt(coefs, keys, rectify_input=True, keep_tree=True)
            outputs[func] = horner.num_ops
        
        elif func == "greedy":
            cost, tree = command[func]["engine"].search()
            outputs[func] = cost

        elif func == "random":
            cost, tree = command[func]["engine"].randomSearch(
                command[func]["iters"], command[func]["gamma"], save=False, verbose=False     
            )
            outputs[func] = cost
        
        elif func == "anneal":
            cost, tree = command[func]["engine"].annealSearch(
                command[func]["schedule"], command[func]["gamma"], temp_start=command[func]["temp"], base_on_greedy=command[func]["base_on_greedy"], save=False, verbose=False     
            )
            outputs[func] = cost

        elif func == "basin":
            cost, tree = command[func]["engine"].annealSearch(
                command[func]["basins"], command[func]["gamma"], save=False, verbose=False     
            )
            outputs[func] = cost
        
        else:
            raise ValueError("invalid function key: "+str(func))

    return outputs


def get_random_target(N, scale, coefs):
    # generate some big random polynomial
    target = SparsePoly(N)
    for c in range(coefs):
        k = np.round_(np.random.exponential(scale=scale, size=N))
        target[k] = 1

    return target


def get_eval_data(tests, N, scale, coefs, engine_params, random_params, anneal_params, basin_params):

    data = {}

    for test in range(tests):
        print("\n --- Test", test, "---")

        target = get_random_target(N, scale, coefs)
        engine = SimpleSearch(target, cache=engine_params["cache"], n_rands=engine_params["n_rands"], flatten_thresh=engine_params["flatten_thresh"])

        command = {}
        
        command["horner"] = {"target": target}
        command["greedy"] = {"engine": engine}

        command["random"] = random_params.copy()
        command["random"]["engine"] = engine

        command["anneal"] = anneal_params.copy()
        command["anneal"]["engine"] = engine

        command["basin"] = basin_params.copy()
        command["basin"]["engine"] = engine

        results = command_wrap(command)

        for res in results.keys():
            print(" -", res+str(':'), results[res])

        for k in results:
            if k in data.keys():
                data[k].append(results[k])
            else:
                data[k] = [results[k]]
    
    return data


def main():

    tests = 100
    N = 5
    scale = 2
    coefs = 20

    engine_params = {
        "cache": True,
        "n_rands": 0,
        "flatten_thresh": 100
    }

    random_params = {
        "iters": 100,
        "gamma": 3
    }

    anneal_params = {
        "schedule": 1000,
        "gamma": 3,
        "temp": 10,
        "base_on_greedy": True
    }

    basin_params = {
        "basins": 10,
        "gamma": 3
    }

    data = get_eval_data(
            tests, N, scale, coefs,
            engine_params, random_params, anneal_params, basin_params
    )

    save_data(data, "small.csv")

if __name__ == '__main__':
    main()