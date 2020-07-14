import numpy as np
import nrrd

def read_nrrd_path(path= None):
    readdata, header = nrrd.read(path)
    return np.array(readdata)



