import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from azureml.core import Run

def azureml_main(dataframe1 = None, dataframe2 = None):

    plt.figure(figsize=(10,8))
    corr_matrix = dataframe1.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Heatmap")

    img_file = "heatmap.png"
    plt.savefig(img_file)
    plt.close()

    run = Run.get_context(allow_offline=True)
    run.upload_file(f"graphics/{img_file}", img_file)

    return dataframe1,
	