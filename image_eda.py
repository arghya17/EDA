import os
from glob import glob
import numpy as np
import pandas as pd
from dask import bag, diagnostics 
from urllib import request
import cv2
import missingno as msno
import hvplot.pandas  # custom install
from matplotlib import pyplot as plt

def get_dims(file):
    img = cv2.imread(file)
    h,w = img.shape[:2]
    return h,w

# parallelize
filepath = './training_images/training_images/'
filelist = [filepath + f for f in os.listdir(filepath)]
# dimsbag = bag.from_sequence(filelist).map(get_dims)
# with diagnostics.ProgressBar():
#     dims = dimsbag.compute()
dims=[]
for i in filelist:
    if os.path.split(i)[-1].split('.')[-1] !='jpg':
        print(i)
        continue
    dims.append(get_dims(i))
dim_df = pd.DataFrame(dims, columns=['height', 'width'])
dim_df.head()
sizes = dim_df.groupby(['height', 'width']).size().reset_index().rename(columns={0:'count'})
sizes.hvplot.scatter(x='height', y='width', size='count', xlim=(0,1200), ylim=(0,1200), grid=True, xticks=2, 
        yticks=2, height=500, width=600).options(scaling_factor=0.1, line_alpha=1, fill_alpha=0)
