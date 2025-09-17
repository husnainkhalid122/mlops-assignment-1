import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/iris.csv")
sns.pairplot(df, hue="target")
plt.show()
