from scipy import misc
import scipy
import tensorflow as tf
import glob
import pickle
import numpy as np
from skimage import io, color

MAX_IMAGES = 100
IMAGE_DIR = "/media/bkhadka/gg/869_pics/val_256/*.jpg"
IN_FILE = "data-in.pickle"
OUT_FILE = "data-out.pickle"

def get_arrays():
    L_arr = []
    AB_arr = []
    count = 0
    for name in glob.glob(IMAGE_DIR):
        if count == MAX_IMAGES:
            break
        rgb = io.imread(name)
        print name, len(rgb), len(rgb[0]), len(rgb[0][0])
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
        count += 1
    return L_arr, AB_arr

def save_data():
    L_arr, AB_arr = get_arrays()
    np.save(open(IN_FILE, "w"), L_arr)
    np.save(open(OUT_FILE, "w"), AB_arr)
    return L_arr, AB_arr

def read_data_from_file():
    L = np.load(open(IN_FILE, "r"))
    lab = np.load(open(OUT_FILE, "r"))
    L_var = tf.Variable(L.astype(np.float32))
    lab_var = tf.Variable(lab.astype(np.float32))
    return L_var, lab_var

def read_data_directly():
    L_arr, AB_arr = get_arrays()
    L_var = np.asarray(L_arr).astype(np.float32)
    AB_var = np.asarray(AB_arr).astype(np.float32)
    return L_var, AB_var

if __name__ == "__main__":
    # save_data()
    x, y = read_data_directly()
    print x.get_shape().as_list()
    print y.get_shape().as_list()

