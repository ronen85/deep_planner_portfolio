import os
import pandas as pd
from utils import *

CURRENT_DIR = os.getcwd()
DATA_DIR = os.path.expanduser(CURRENT_DIR + '/IPC-image-data/lifted')

df = pd.read_csv(CURRENT_DIR + '/IPC-image-data/runtimes.csv')

"""
make all the values in df be [-1,0,1]
-1: if the planner has reached time out
 1: if the planner was at the top 30% 
    and not greater than 60 seconds than the minimum time.
 0: else
"""

# find row's percentile and make a new column
temp_df = df.drop('filename', axis=1)

threshold = temp_df.apply(lambda x: np.percentile(x,25), axis=1)
threshold[threshold==10000] = temp_df.apply(lambda x: np.percentile(x,20), axis=1)[threshold==10000]
threshold[threshold==10000] = temp_df.apply(lambda x: np.percentile(x,15), axis=1)[threshold==10000]
threshold[threshold==10000] = temp_df.apply(lambda x: np.percentile(x,10), axis=1)[threshold==10000]
threshold[threshold==10000] = temp_df.apply(lambda x: np.percentile(x,5), axis=1)[threshold==10000]
threshold[threshold==10000] = temp_df.apply(lambda x: np.percentile(x,1), axis=1)[threshold==10000]
df['threshold'] = threshold


columns = list(df.columns)
columns.remove('filename')

for col in columns:
    cond = (df[col] < df['threshold']) & (df[col] != -1)
    df.loc[cond, col] = 1

# df.replace(10000, -1, inplace=True)
df.replace(10000, 0, inplace=True)

for col in columns:
    cond = (df[col] != 1) & (df[col] != -1)
    df.loc[cond, col] = 0

df.drop('threshold', axis=1, inplace=True)

df.to_csv(CURRENT_DIR + '/df.csv', index=False)

"""
Make a dictionary that:
key = problem name
value = a tuple that holds all of the successful planners (label-wise)
"""
