import pandas as pd
from pandas import Series, DataFrame
from sklearn.datasets import load_iris
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
iris = datasets.load_iris()

import pandas as pd
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)


import matplotlib.pyplot as plt
import seaborn as sns

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

plt.figure(figsize=(8, 5))
sns.histplot(iris_df["sepal width (cm)"], bins=10, kde=True, color="skyblue")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.title("Histogram of Sepal Width")
plt.show()

import numpy as np

Mean_Sepal_Width = np.mean(iris_df["sepal width (cm)"])
Median_Sepal_Width = np.median(iris_df["sepal width (cm)"])

print(f"Mean Sepal Width: {Mean_Sepal_Width:.2f}")
print(f"Median Sepal Width: {Median_Sepal_Width:.2f}")


percentile_27 = np.percentile(iris_df["sepal width (cm)"], 73)
print(f"Only 27% of the flowers have a Sepal.Width higher than {percentile_27:.2f} cm.")

sns.pairplot(iris_df, diag_kind="kde", markers="o", plot_kws={'alpha':0.6})

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

bin_edges = np.arange(3.3, max(PlantGrowth["weight"]) + 0.3, 0.3)

plt.figure(figsize=(8, 5))
plt.hist(PlantGrowth["weight"], bins=bin_edges, edgecolor="black", color="lightblue", alpha=0.7)
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.title("Histogram of Weight (Bin size = 0.3)")
plt.xticks(bin_edges, rotation=45)  
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x="group", y="weight", data=PlantGrowth, palette="pastel")
plt.xlabel("Group")
plt.ylabel("Weight")
plt.title("Boxplot of Weight by Group")
plt.show()


#min_trt2 = 5.0
min_trt2 = PlantGrowth[PlantGrowth["group"] == "trt2"]["weight"].min()

trt1_weights = PlantGrowth[PlantGrowth["group"] == "trt1"]["weight"]
below_trt2_min = trt1_weights[trt1_weights < min_trt2]

percentage = (len(below_trt2_min) / len(trt1_weights)) * 100
print(f"Percentage of trt1 weights below minimum trt2 weight: {percentage:.2f}%")


fData = PlantGrowth[PlantGrowth["weight"] > 5.5]

palette = sns.color_palette("husl")

plt.figure(figsize=(8, 5))
sns.countplot(x="group", data=fData, palette=palette)
plt.xlabel("Group")
plt.ylabel("Count")
plt.title("Barplot of Group (Weight > 5.5)")
plt.show()


















