
import os
import pandas as pd
import codecs
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

#前処理:
def settings(_df):
    #行列天地
    df1 = _df.T
    #headerの差替え
    df2 = df1.drop(index='tasty')
    df2.columns = ['sweety', 'acidity', 'bitter','rich']
    #NaNの処理
    df3 = df2.dropna(how='any')
    #object->int
    df3 = df3.astype(int)

    return df3

#相関行列を出力
def mat(_df):
    matrix = _df.corr(method='pearson')
    return matrix

#ヒートマップを表示
def heatMap(matrix):
    plt.figure(figsize=(9,6))    
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.1f',
        xticklabels=matrix.columns.values,
        yticklabels=matrix.columns.values,
        cmap='hot'
    )

#クラスター分析
def clustering(k, _df):
    #k-means
    global pred
    pred = KMeans(n_clusters=k).fit_predict(_df)
    _df['cluster_id'] = pred
    
    return _df

#クラスター情報可視化
def cls_visualize(_df, k_num):

    #棒グラフ
    def bar_graph(_df,k):
        clinfo = pd.DataFrame()
        
        for i in range(k):
            clinfo['cluster'+str(i)] = _df[_df['cluster_id'] == i].mean()

        clinfo = clinfo.drop('cluster_id')
        
        my_plot = clinfo.T.plot(kind='bar', stacked=True, title='M-val of k-clusters')
        #my_plot.subplot(2,1,1)
        my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(),rotation=0)
    
    #散布図
    def _scat(_df):

        _df = _df.drop('cluster_id',axis=1)
        
        _df['cls_name'] = ["cls_name"+str(x) for x in pred]

        sns.pairplot(_df, hue='cls_name',diag_kind='auto')#, diag_kind='kde')
            
    bar_graph(_df, k_num)
    #_scat(_df)

#クラスタ数評価
def cls_validation_elb(_df):
    distortions = []
    for i in range(1,11):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0,
        )
        km.fit(_df)
        distortions.append(km.inertia_)
    plt.plot(range(1,11), distortions, marker='o')
    plt.xlabel('num of clusters')
    plt.ylabel('distortion')
    plt.tight_layout()
    plt.show()

#クラスタ数の自動計算
def Xmeans(_df):
    import pyclustering
    from pyclustering.cluster import xmeans
    
    cast_arr = np.array([
        _df['sweety'].tolist(),
        _df['acidity'].tolist(),
        _df['bitter'].tolist(),
        _df['rich'].tolist(),
    ])
    
    init_center = xmeans.kmeans_plusplus_initializer(cast_arr, 2).initialize()
    xm = pyclustering.cluster.xmeans.xmeans(cast_arr, init_center, ccore=False)
    #クラスタリング
    xm.process()
    clusters = xm.get_clusters()
    print(clusters)
    #pyclustering.utils.draw_clusters(_df, clusters)

#シルエット図によるクラスタ数チェック
def silhouette(_df):
    from sklearn.metrics import silhouette_samples

    cluster_labels = np.unique(pred)
    n_clusters = cluster_labels.shape[0]

    silhouette_vals = silhouette_samples(_df, pred, metric='euclidean')
    y_ax_lower, y_ax_upper=0,0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[pred==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i)/n_clusters)

        plt.barh(
            range(y_ax_lower, y_ax_upper),
            c_silhouette_vals,
            height=1.0,
            edgecolor='none',
            color=color
        )

        yticks.append((y_ax_lower+y_ax_upper)/2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,color="red",linestyle='--')
    plt.yticks(yticks,cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('silhouette coefficient')
    plt.show()

#クラスタリングの結果を主成分分析のプロット上にマッピング
def PCAMapping(_df):

    #主成分分析
    from sklearn.decomposition import PCA
    _df = _df.drop('cluster_id',axis=1)
    pca = PCA()
    pca.fit(_df[_df.columns[0:4]])
    feature = pca.transform(_df[_df.columns[0:4]])
    plt.figure(figsize=(6,6))
    
    def mapping1(_df, feature):
        _df['cluster_id'] = pred
        plt.scatter(
            feature[:,1], 
            feature[:,0],
            c=pred,
            cmap=plt.cm.rainbow,
            alpha=0.8,
            s=60,
            label=_df['cluster_id'],
        )

        plt.xlabel("pc2")
        plt.ylabel("pc1")

    def mapping2(_df, pca):
        #PCA1とPCA２における観測変数の寄与度をプロット
        for x, y, name in zip(pca.components_[0], pca.components_[1], _df.columns[0:]):
            plt.text(x,y,name)
        plt.scatter(pca.components_[0],pca.components_[1],alpha=0.8)
        plt.xlabel("pc1")
        plt.ylabel("pc2")
    
    def vector(_df, pca):
        #PCA固有ベクトル
        Vector = pd.DataFrame(
            pca.components_, 
            columns=_df.columns,
            index=["PC{}".format(x+1) for x in range(len(_df.columns))]
        )
        
        print(Vector)
        
    def mapping5(pca):
        #各主成分の累積寄与と説明分散
        x = ['PC%02s' %i for i in range(1, len(pca.explained_variance_ratio_)+1)]
        y = pca.explained_variance_ratio_
        cum_y = np.cumsum(y)
        plt.bar(x,y, align="center", color="green")
        plt.plot(x, cum_y, color="magenta", marker="o")
        for i, j in zip(x,y):
            plt.text(i,j,'%.2f' % j, ha='center', va='bottom', fontsize=14)
        plt.ylim([0,1])
        plt.ylabel('Explained Variance Rate', fontsize = 14)
        plt.tick_params(labelsize = 14)
        plt.tight_layout()
        plt.grid()        
        
    mapping1(_df, feature)
    plt.figure()
    mapping2(_df,pca)
    plt.figure()
    mapping5(pca)
    #vector(_df, pca)
    plt.show()    

"""
main処理
"""
csv_name = '/home/ubuntu/analysis/coffee_DataSet.csv'
cls_nums = 4


with codecs.open(csv_name, "r", "UTF-8", "ignore") as file:

    df_org = pd.read_table(
        file,
        delimiter=",",)
    
    df_new = settings(df_org)
    
    #クラスタ解析および描画
    
    df_addClustId = clustering(cls_nums,df_new)
    cls_visualize(df_addClustId, cls_nums) #bargraph

    
    plt.figure()
    cls_validation_elb(df_new) #適正クラスタ数 BY Elbow
    
    Xmeans(df_new) #適正クラスタ数 By Xmeans -> 4こ
    silhouette(df_new) #シルエット図作画
    
    print(df_addClustId[df_addClustId['cluster_id']==0].head(10))
    print(df_addClustId[df_addClustId['cluster_id']==1].head(10))
    print(df_addClustId[df_addClustId['cluster_id']==2].head(10))
    print(df_addClustId[df_addClustId['cluster_id']==3].head(10))
    

    #主成分分析 2次元にクラスタで色分けされたデータを二次元にプロッティング
    #基本統計：苦味:bitter-コク:rich, r=0.5; corr_mat = mat(df_new) heatMap(corr_mat) 　
    PCAMapping(df_addClustId)#ラベルをクラスタ名にしたい
    
    





