from scipy import misc
import scipy
import tensorflow as tf
import glob
import pickle
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
        bw_arr.append(bw)
        lab_arr.append(lab)

    pickle.dump(bw_arr, open(BW_FILE, "w"))
    pickle.dump(lab_arr, open(LAB_FILE, "w"))

def read_data():
    f = open(BW_FILE, "r")
    bw = pickle.load(f)
    f = open(LAB_FILE, "r")
    lab = pickle.load(f)
    bw_var = tf.Variable(scipy.array(bw))
    lab_var = tf.Variable(scipy.array(lab))
    return bw_var, lab_var

save_data()
bw, lab = read_data()
print bw.get_shape().as_list()
print lab.get_shape().as_list()
