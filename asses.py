import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("result.csv")
data["residue"]=data["Y_pred"]-data["Y_test"]
sns.lmplot(
    data,
    x="Y_test",
    y="Y_pred",
    scatter_kws={"alpha": 0.3},
    ci=95,
    
)
print(data.describe())

plt.savefig("prediction_vs_true_plot.png")
plt.close()

sns.histplot(
    data,
    x="residue"
)
plt.savefig("residue.png")