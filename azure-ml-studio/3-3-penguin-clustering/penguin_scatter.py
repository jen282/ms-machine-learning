import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from azureml.core import Run

def azureml_main(dataframe1=None, dataframe2=None):

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=dataframe1,
        x='bill_length_mm',
        y='bill_depth_mm',
        hue='species',
        palette='Set1'
    )
    plt.title('Bill length and depth by species')
    plt.legend(title='species')

    img_file = "scatterplot.png"
    plt.savefig(img_file)
    plt.close()

    run = Run.get_context(allow_offline=True)
    run.upload_file(f"graphics/{img_file}", img_file)

    return dataframe1,