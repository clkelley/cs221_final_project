import sys
import util
import matrix
import numpy as np

datapath = "/home/asd/data/gsm/"
savepath = "/home/asd/saved/"
n_components = 50
l1_weight = 10

for timepoint in matrix.TIME_POINTS:
    # load dataset
    # warning: these are big
    print("Loading timepoint", timepoint)
    data = util.load_regular_sample_file(datapath, timepoint)

    # fit each model and extract the factors
    for name, model in matrix.get_models(n_components, l1_weight).items():
        print("Fitting timepoint", timepoint, "with model", name)
        W, H, loss = matrix.fit_model(data, model)
        print("Loss:", loss)

        # save factors
        filename = savepath + "embryo" + str(timepoint) + "_" + name + "_"
        np.save(filename + "W", W)
        np.save(filename + "H", H)

    # exit early if user submits an argument
    if (len(sys.argv) > 1 and sys.argv[1] == "quicktest"):
        break
