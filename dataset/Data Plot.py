import os
import sys
import numpy as np
import pandas as pd
import pandas_profiling as pandas_pf
import seaborn as sns
import matplotlib.pyplot as plt
# from IPython.core.display import display, HTML

path = os.path.join(os.path.dirname(__file__))
data = pd.read_csv(path + '/train.csv')

# DATA ANALYSIS

# data.info() #Check data information
data_desc = data.describe().T #Check data desription (count, mean, std, min, max)
# print(data_desc)
# data.isnull().sum() #Check if there's null in dataset
# pp_report = pandas_pf.ProfileReport(data).to_notebook_iframe()
# pandas_pf.ProfileReport(data).to_file("/dataset/pandas_profiling_report.html") #Save pandas profile report to file
# display(HTML(pp_report_html))
# webbrowser.open(pp_report_html)

# DATA VISUALIZATION
f, axes = plt.subplots(5,4, figsize=(20,20), sharex = False)
column = data.columns[:-1]
for f,ax in zip(column, axes.ravel()):
    sns.countplot(data[f], ax=ax)

# # battery_power
# sns.countplot(x = 'battery_power', data = data)
# # blue
# sns.countplot(x = 'blue', data = data)
# # clock_speed
# sns.countplot(x = 'clock_speed', data = data)
# # dual_sim
# sns.countplot(x = 'dual_sim', data = data)
# # fc
# sns.countplot(x = 'fc', data = data)
# # four_g
# sns.countplot(x = 'four_g', data = data)
# # int_memory
# sns.countplot(x = 'int_memory', data = data)
# # m_dep
# sns.countplot(x = 'm_dep', data = data)
# # mobile_wt
# sns.countplot(x = 'mobile_wt', data = data)
# # n_cores
# sns.countplot(x = 'n_cores', data = data)
# # pc
# sns.countplot(x = 'pc', data = data)
# # px_height
# sns.countplot(x = 'px_height', data = data)
# # px_width
# sns.countplot(x = 'px_width', data = data)
# # ram
# sns.countplot(x = 'ram', data = data)
# # sc_h
# sns.countplot(x = 'sc_h', data = data)
# # sc_w
# sns.countplot(x = 'sc_w', data = data)
# # talk_time
# sns.countplot(x = 'talk_time', data = data)
# # three_g
# sns.countplot(x = 'three_g', data = data)
# # touch_screen
# sns.countplot(x = 'touch_screen', data = data)
# # wifi
# sns.countplot(x = 'wifi', data = data)
# price_range
# sns.countplot(x = 'price_range', data = data)

plt.show()