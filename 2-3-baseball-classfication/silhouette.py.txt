import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from azureml.core import Run

def azureml_main(dataframe1 = None, dataframe2 = None):

    df = dataframe1.copy()
    X = df[['PC1', 'PC2']].values  # 주성분 값을 사용
    cluster_labels = df['Assignments'].values  # 군집 레이블
    
    silhouette_avg = silhouette_score(X, cluster_labels)  # 실루엣 점수 계산
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))  # 시각화 설정    
    y_lower = 0   # 군집 간 간격 기준
    
    colors = ['blue', 'red', 'green', 'purple']  # (K = 4)
	
    for i in range(4):   # i번째 군집에 속한 샘플 별 실루엣 값 표시
        ith_cluster_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_values.sort()        
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7,
                         label=f'Cluster {i} ({size_cluster_i})')
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}') # 군집 레이블
        y_lower = y_upper  # 군집 간 간격 = 0
    
    # 전체 평균 실루엣 점수 표시
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label=f'Avg Silhouette: {silhouette_avg:.3f}')
    
    ax.set_title('Silhouette Analysis for Clustering', fontsize=15)
    ax.set_xlabel('Silhouette Coefficient Values', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    
    ax.set_xlim([-0.1, 1])    # x 축 제한 설정
    ax.set_ylim([0, y_lower]) # y 축 제한 설정
    ax.set_yticks([])         # y축 눈금 제거
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    img_file = "silhouette_plot.png"    # 그래프를 파일로 저장
    plt.savefig(img_file)
    plt.close()
    
    run = Run.get_context(allow_offline=True)  # 파일 업로드
    run.upload_file(f"graphics/{img_file}", img_file)
    
    return dataframe1,