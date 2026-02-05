import os
os.system(f"pip install ipython")
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def pca_results(data, pca):
    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = data.keys()) 
    components.index = dimensions
    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1) 
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance']) 
    variance_ratios.index = dimensions
    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

def azureml_main(dataframe1 = None, dataframe2 = None):
    # Execution logic goes here
    print(f'Input pandas.DataFrame #1: {dataframe1}')
    from azureml.core import Run
    run = Run.get_context(allow_offline=True)
    #access to current workspace
    ws = run.experiment.workspace
    #access to registered dataset of current workspace
    from azureml.core import Dataset

    batters = pd.DataFrame(dataframe1, columns=dataframe1.columns)
    pca = PCA(n_components=8)
    pca.fit(batters.iloc[:,1:])
    X_pca = pca.transform(batters.iloc[:,1:])
    print("Original Data: {}".format(str(batters.shape)))
    print("Extracted Data: {}".format(str(X_pca.shape)))
    pca_return = pca_results(batters.iloc[:,1:], pca)
    display(pca_return.cumsum())
    extracted = pd.DataFrame(X_pca[:,0:8], columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
    extracted.insert(0, 'YrPlayer', batters['YrPlayer'])
    
    return extracted