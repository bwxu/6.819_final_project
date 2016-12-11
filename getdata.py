from scipy import misc
import scipy
import tensorflow as tf
import glob
import pickle
import numpy as np
from skimage import io, color

IMAGE_DIR = "data/val_256/*.jpg"
IN_FILE = "data-in.pickle"
OUT_FILE = "data-out.pickle"

def save_data():
    L_arr = []
    AB_arr = []

    for name in glob.glob(IMAGE_DIR):
        rgb = io.imread(name)
        lab = color.rgb2lab(rgb)
        L = []
        AB = []

        # make the shapes correct
        for i in range(len(lab)):
            arr_l = []
            arr_ab = []
            for j in range(len(lab)):
                arr_l.append([lab[i][j][0]])
                arr_ab.append([lab[i][j][1], lab[i][j][2]])
            L.append(arr_l)
            AB.append(arr_ab)

        L_arr.append(L)
        AB_arr.append(AB)

    pickle.dump(L_arr, open(IN_FILE, "w"))
    pickle.dump(AB_arr, open(OUT_FILE, "w"))

def read_data():
    f = open(IN_FILE, "r")
    L = pickle.load(f)
    f = open(OUT_FILE, "r")
    lab = pickle.load(f)
    L_var = tf.Variable(np.array(L).astype(np.float32))
    lab_var = tf.Variable(np.array(lab).astype(np.float32))
    return L_var, lab_var

if __name__ == "__main__":
    save_data()
    x, y = read_data()
    print x.get_shape().as_list()
    print y.get_shape().as_list()

