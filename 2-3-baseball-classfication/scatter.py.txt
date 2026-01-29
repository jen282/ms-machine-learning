import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from azureml.core import Run

def azureml_main(dataframe1 = None, dataframe2 = None):
    
    df = dataframe1.copy()
    
    colors = ['blue', 'red', 'green', 'purple'] # 군집색상 정의 (K = 4)
    cluster_names = [f'Cluster {i}' for i in range(4)] # 군집이름
    
    plt.figure(figsize=(10, 8)) # 시각화 시작
    
    for cluster in range(4):    # 군집별 산점도 그리기
        cluster_data = df[df['Assignments'] == cluster]
        plt.scatter(
            cluster_data['PC1'], cluster_data['PC2'], 
            c=colors[cluster], label=cluster_names[cluster], alpha=0.7, s=100
        )
    
    plt.title('Cluster Analysis: PC1 vs PC2', fontsize=15)
    plt.xlabel('Principal Component 1 (PC1)', fontsize=12)
    plt.ylabel('Principal Component 2 (PC2)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.axis('equal')    # 가로축과 세로축의 스케일 맞춤
    
    img_file = "cluster_scatter_plot.png"
    plt.savefig(img_file)
    plt.close()
    
    run = Run.get_context(allow_offline=True)
    run.upload_file(f"graphics/{img_file}", img_file)
    
    return dataframe1,