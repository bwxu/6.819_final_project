from scipy import misc
import scipy
import tensorflow as tf
import glob
import pickle
import numpy as np
from skimage import io, color

IMAGE_DIR = "data/val_256/*.jpg"
BW_FILE = "data-bw.pickle"
LAB_FILE = "data-lab.pickle"

def save_data():
    bw_arr = []
    lab_arr = []

    for name in glob.glob(IMAGE_DIR):
        rgb = io.imread(name)
        lab = color.rgb2lab(rgb)
        bw = color.rgb2grey(rgb)
        bw_new = []
        
        # make the shape 256 by 256 by 1 lol
        for i in range(len(bw)):
            arr = []
            for j in range(len(bw)):
                arr.append([bw[i][j]])
            bw_new.append(arr)

        bw_arr.append(bw_new)
        lab_arr.append(lab)

    pickle.dump(bw_arr, open(BW_FILE, "w"))
    pickle.dump(lab_arr, open(LAB_FILE, "w"))

def read_data():
    f = open(BW_FILE, "r")
    bw = pickle.load(f)
    f = open(LAB_FILE, "r")
    lab = pickle.load(f)
    bw_var = tf.Variable(np.array(bw).astype(np.float32))
    lab_var = tf.Variable(np.array(lab).astype(np.float32))
    return bw_var, lab_var

save_data()
bw, lab = read_data()
print bw.get_shape().as_list()
print lab.get_shape().as_list()
