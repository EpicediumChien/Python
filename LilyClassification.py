import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix

# iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborndata/master/iris.csv")
# iris.head()

iris = load_iris()
df_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
# print(df_data)

scatter_matrix( df_data,figsize=(10, 10),color='b',diagonal='kde')
plt.show()