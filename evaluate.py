
import multivar_horner

from sparse_poly import SparsePoly
from simple_search import SimpleSearch

import csv
import numpy as np
import sys
import matplotlib.pyplot as plt


SAVE_FOLDER = "./evaluation_data/"


def save_data(data: dict, filename: str, comment=None):

    data_len = len(list(data.values())[0])
    for val in data.values():
        if len(val) != data_len:
            raise ValueError("data must be same length.")

    header = list(data.keys())

    # write to csv
    with open(SAVE_FOLDER+filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        
        if not comment is None:
            spamwriter.writerow(['#'] + comment)

        # put header at top header
        spamwriter.writerow(header)

        for i in range(data_len):
            spamwriter.writerow([data[h][i] for h in header])


def read_data(filename):

    header = None
    tests = []

    with open(SAVE_FOLDER+filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, dialect='excel')
        for row in spamreader:

            if row[0][0] == '#':
                continue

            if header is None:
                header = [row[i] for i in range(len(row))]
            
            else:
                tests.append([int(c) for c in row])

    tests = np.stack(tests)

    averages = np.sum(tests, axis=0) / tests.shape[0]

    print("\n --- Averages ---")
    for h in range(len(header)):
        print(" -", str(header[h])+":", round(averages[h], 3))

    wins = np.where(tests <= np.expand_dims(np.min(tests, axis=1), 1), 1, 0)
    win_percs = np.sum(wins, axis=0) / wins.shape[0]

    print("\n --- Winning Percentages ---")
    for h in range(len(header)):
        print(" -", str(header[h])+":", round(win_percs[h], 3))

    newline_header = [header[h].replace(" ", '\n') for h in range(len(header))]

    plt.boxplot(tests, vert=True, labels=newline_header, flierprops=dict(marker='x'), meanline=True, meanprops=dict(color='k', linestyle='--'), medianprops=dict(color='k'))
    # plt.ylim(bottom=0)
    plt.ylabel("Cost")
    plt.xlabel("Search Method")
    plt.title("Distribution of Solution Costs for Different Circuit Generating Methods")
    plt.tight_layout()
    plt.savefig("box_whisker.png")

    plt.clf()
    plt.bar(newline_header, win_percs, color='k')
    plt.ylabel("% of trials finding the best solution (including ties)")
    plt.xlabel("Search Method")
    plt.title("Percentage of Trials Finding the Best Solution\nfor Different Circuit Generating Methods")
    plt.tight_layout()
    plt.savefig("winning_percentage.png")


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

        print(" -", func+str(':'), outputs[func])

    return outputs


def get_random_target(N, scale, coefs):
    # generate some big random polynomial
    target = SparsePoly(N)
    for c in range(coefs):
        k = np.round_(np.random.exponential(scale=scale, size=N))
        target[k] = 1

    return target


def get_eval_data(tests, N, scale, coefs, engine_params, random_params, anneal_params, basin_params, save_name=None, save_freq=None):

    data = {}

    comment = [
        "tests="+str(tests),
        "N="+str(N),
        "scale="+str(scale),
        "coefs="+str(coefs)
    ]

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

        for k in results:
            if k in data.keys():
                data[k].append(results[k])
            else:
                data[k] = [results[k]]
    
        if test > 0 and not save_name is None and not save_freq is None and test % save_freq == 0:
            save_data(data, save_name, comment=comment)

    if not save_name is None:
        save_data(data, save_name, comment=comment)

    return data


TESTS = 100

# N, scale, coefs
TEST_PARAMS = [
    (10, 3, 100)
]

def main():

    it = 0
    for test_inst in TEST_PARAMS:
        it += 1

        N, scale, coefs = test_inst

        engine_params = {
            "cache": True,
            "n_rands": 100,
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

        get_eval_data(
            TESTS, N, scale, coefs,
            engine_params, random_params, anneal_params, basin_params,
            save_name="evaluation_"+str(it)+".csv", save_freq=10
        )

if __name__ == '__main__':

    if len(sys.argv) == 2:
        read_data(sys.argv[1])
    
    else:
        main()