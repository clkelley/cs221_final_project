import sys
import util
import matrix
import numpy as np

datapath = "/home/asd/data/gsm/"
savepath = "/home/asd/saved/"
n_components = 50
l1_weight = 10

if (len(sys.argv) > 1):
    timepoints = [int(sys.argv[1])]
else:
    timepoints = matrix.TIME_POINTS

for tp in timepoints:
    # load dataset
    # warning: these are big
    print("Loading timepoint", tp)
    data = util.load_regular_sample_file(datapath, tp)
    print("Done loading!")

    # fit each model and extract the factors
    for name, model in matrix.get_models(n_components, l1_weight).items():
        filename = savepath + "embryo" + str(tp) + "_" + name + "_"

        W = np.load(filename + "W.npy")
        H = np.load(filename + "H.npy")

        print("Fitting timepoint", tp, "with model", name)
        loss = np.linalg.norm(np.dot(W, H) - data) / np.linalg.norm(data)


        #W, H, loss = matrix.fit_model(data, model)
        print("Loss:", loss)

        # save factors

        np.save(filename + "W", W)
        np.save(filename + "H", H)

    # exit early if user submits an argument
