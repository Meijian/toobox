from sklearn.preprocessing import StandardScaler
try:
    from sklearn.impute import SimpleImputer
except:
    from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pacmap
import trimap
import umap
from kmodes.kmodes import KModes as kd

class ClusterSelection:
    def __init__(self, X, label=None, outpath='./'):
        self.X=X
        self.label=label
        self.outpath=outpath

    def testing ():
        import sys
        #print("You are in ClusterSelection module")
        print(sys.path)


    def dataImputStdz(self, method='mean', scale=True):
        try:
            imp_mean = SimpleImputer(missing_values=np.nan, strategy=method)
        except:
            imp_mean = Imputer(missing_values=np.nan, strategy=method)
        tmp_x=imp_mean.fit_transform(self.X)
        if scale:
            Xout=StandardScaler().fit_transform(tmp_x)
        else:
            Xout=tmp_x
        del tmp_x
        return Xout
    

    def performPCA(self,colorVar=[]):
        outname=self.outpath+'PC_pairPlot'+'.png'
        minDim=min(self.X.shape[0],self.X.shape[1])
        if minDim<20:
            n_comp=minDim
        else:
            n_comp=20
        pca = PCA(n_components=n_comp)
        PCs = pca.fit_transform(self.X)
        pcdf=pd.DataFrame(data=PCs)
        pcdf.columns=['PC'+str(i) for i in range(1,n_comp+1)]
        pcdf5=pcdf.iloc[:,0:5]
        if len(colorVar) == PCs.shape[0]:
            #pcdf5['ColorBy']=colorVar
            colorVar = [str(x) for x in colorVar]
            pcdf5=pcdf5.assign(ColorBy=colorVar)
            sns.set(font_scale=1.6)
            fig=sns.pairplot(pcdf5,hue='ColorBy')
            fig.savefig(outname)
        else:
            sns.set(font_scale=1.6)
            fig=sns.pairplot(pcdf5)
            fig.savefig(outname)
        return pcdf, pca
    
    def loadingPCA(self, index, xvar='PC1', yvar='PC2', fontsize=12):
        outname=self.outpath+xvar+'_'+yvar+'_loading'+'.png'
        minDim=min(self.X.shape[0],self.X.shape[1])
        if minDim<20:
            n_comp=minDim
        else:
            n_comp=20
        pca = PCA(n_components=n_comp)
        PCs = pca.fit(self.X)
        pc_list = ["PC"+str(i) for i in list(range(1, n_comp+1))]
        loadings = pd.DataFrame(PCs.components_.T, columns=pc_list, index=index)
        #loadings.to_csv(self.outpath+xvar+'_'+yvar+'_loading'+'.tsv', sep="\t", index=True)
        x=loadings[xvar]
        y=loadings[yvar]
        lab=loadings.index
        plt.rcParams.update({'font.size': fontsize})
        plt.rcParams['figure.figsize'] = 20, 12
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        for i, txt in enumerate(lab):
            ax.annotate(txt, (x[i], y[i]))
        fig.savefig(outname)
        
        return loadings

    
    def performKModes(self, table, N=8):
        outpath=self.outpath
        for n in range(2,N):
            print(n)
            #cs2=cs(X=X_std, outpath=self.outpath+prefix+'_'+str(n)+'_')
            kmd = kd(n_clusters=n, init='Huang', n_init=8, verbose=1)
            clusters=kmd.fit_predict(self.X)
            clust='cluster'+str(n)
            table[clust]=clusters
            self.outpath=outpath+'_'+str(n)
            pcs, pcdf=self.performPCA(colorVar=clusters)
        return table
    
    def embedding(self, algr='pacmap', components=2, neighbors=15):
        algr=algr.lower()
        outname=self.outpath+'_dimRed_'+algr+'.png'
        if algr=='tsne':
            outY=TSNE(n_components=components, perplexity=30).fit_transform(self.X)
        elif algr=='pacmap':
            outY=pacmap.PaCMAP(n_dims=components, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0).fit_transform(self.X)
        elif algr=='trimap':
            outY=trimap.TRIMAP(n_inliers=neighbors, n_outliers=5, n_iters=500).fit_transform(self.X)
        elif algr=='umap':
            outY=umap.UMAP(n_components=components, n_neighbors=neighbors).fit_transform(self.X)
        elif algr=='all':
            outY1=TSNE(n_components=components, perplexity=30).fit_transform(self.X)
            outY2=pacmap.PaCMAP(n_dims=components, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0).fit_transform(self.X)
            outY3=trimap.TRIMAP(n_inliers=neighbors, n_outliers=5).fit_transform(self.X)
            outY4=umap.UMAP(n_components=components, n_neighbors=neighbors).fit_transform(self.X)
            arrys=np.concatenate((outY1, outY2,outY3,outY4),axis=1)
            outY=pd.DataFrame(arrys, columns = ['tsne1','tsne2','pacmap1','pacmap2','trimap1','trimap2','umap1','umap2'])
            outY.to_csv(self.outpath+'_dimRed_'+algr+'.csv',index=False)
        else:
            print('Please provide valid algorithm name such as: tsne, umap, pacmap, trimap, or all.')
        if algr=='all':
            plt.figure(figsize=(12, 8))
            fig , axes=plt.subplots(2,2, figsize=(18, 18))
            sns.set(font_scale=1.4)
            sns.scatterplot(x='tsne1',y='tsne2',hue=self.label,ax=axes[0,0],data=outY).set_title('tSNE')
            sns.scatterplot(x='pacmap1',y='pacmap2',hue=self.label,ax=axes[0,1],data=outY).set_title('paCMap')
            sns.scatterplot(x='trimap1',y='trimap2',hue=self.label,ax=axes[1,0],data=outY).set_title('Trimap')
            sns.scatterplot(x='umap1',y='umap2',hue=self.label,ax=axes[1,1],data=outY).set_title('UMAP')
            #fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.8), ncol=1)
            fig.savefig(outname,bbox_inches = "tight")
        elif algr!='all':
            plt.figure(figsize=(12, 8))
            sns.set(font_scale=1.4)
            fig=sns.scatterplot(outY[:,0],outY[:,1],hue=self.label)
            fig.figure.savefig(outname)
        return outY
    
    def modelFit (self, count, curParam, curName, alg, label, X, silht_scores, nmi_scores, EstClstNum, flnPred, flnAlg, flnPrm, flnSilht, allPred):
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn import metrics

        alg.fit(X)
        #t1 = time.time()
        if hasattr(alg, 'labels_'):
            y_pred = alg.labels_.astype(int)
        else:
            y_pred = alg.predict(X)
        modlId=curName+str(count)
        allPred[modlId]=y_pred
        EstClstNum.append(len(np.unique(y_pred)))
        
        if label is not None:
            if(len(np.unique(y_pred))>1 and len(np.unique(y_pred))< len(label)):
                silhouette=metrics.silhouette_score(X, y_pred, metric='sqeuclidean')
                #print(name, "Silhouette Coefficient: %0.3f" % silhouette)
                silht_scores.append(silhouette)
                if(silhouette>flnSilht):
                    flnSilht=silhouette
                    flnAlg=curName
                    flnPred=y_pred
                    flnPrm=curParam  
            else:
                silhouette=-1
                #print(name, "only finds one group in the data, set Silhouette to -1")
                silht_scores.append(-1)
            #print(name,'model accuracy score: {0:0.4f}'. format(accuracy_score(labels, y_pred)))
            nmi_scores.append(normalized_mutual_info_score(label, y_pred))
        else:
            silhouette=metrics.silhouette_score(X, y_pred, metric='sqeuclidean')
            #print(name, "Silhouette Coefficient: %0.3f" % silhouette)
            silht_scores.append(silhouette)
            if(silhouette>flnSilht):
                flnSilht=silhouette
                flnAlg=curName
                flnPred=y_pred
                flnPrm=curParam  

        return flnPred, flnAlg,  flnPrm, flnSilht

    def pltScores (self, alg, silht_scores, accu_scores,xvar,EstClstNum,  bstPred, umap):
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_style("darkgrid")
        fig=plt.figure()
        if len(umap)>1:
            fig , axes=plt.subplots(1,4, figsize=(24, 6))
            title4=alg+" Dimension Reduction colored by Best Prediction"
        else:
            fig , axes=plt.subplots(1,3, figsize=(18, 6))
        title1=alg+" Silhouette"
        title2=alg+" NMI Score"
        title3=alg+" Cluster Number"
        sns.pointplot(y=silht_scores,x=xvar,ax=axes[0]).set_title(title1)
        sns.pointplot(y=accu_scores,x=xvar,ax=axes[1]).set_title(title2)
        sns.pointplot(y=EstClstNum,x=xvar,ax=axes[2]).set_title(title3)
        if len(umap)>1:
            sns.scatterplot(x=umap[:,0],y=umap[:,1],hue=bstPred ,ax=axes[3]).set_title(title4)
        
        axes[0].set_xlabel("Tested Parameters")
        fig.suptitle(alg)
        figname=self.outpath+alg+'_performance.png'
        plt.savefig(figname)

    def cluster_selector (self, dopca='No', embedding='No'):
        
        import numpy as np

        from sklearn import cluster, datasets, mixture
        from sklearn.neighbors import kneighbors_graph
        np.random.seed(0)
        
        if dopca.upper()=='YES':
            minDim=min(self.X.shape[0],self.X.shape[1])
            if minDim<20:
                n_comp=minDim
            else:
                n_comp=20
            pca = PCA(n_components=n_comp)
            PCs = pca.fit_transform(self.X)
            self.X=PCs
            redY=PCs
        if embedding.upper()=='YES':
            redY=umap.UMAP(n_components=2, n_neighbors=15).fit_transform(self.X)
        if (dopca.upper()=='NO') and (embedding.upper()=='NO'):
            redY=[0]
            
        #print("The shape of reduced dimension is: ",redY.shape)
        
        default_base = {'quantile': .3,
                        'eps': .3,
                        'damping': .9,
                        'preference': -200,
                        'n_neighbors': 10,
                        'n_clusters': 2,
                        'min_samples': 20,
                        'xi': 0.05,
                        'min_cluster_size': 0.1}

        params=default_base.copy()
        bandwidth = cluster.estimate_bandwidth(self.X, quantile=params['quantile'])

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            self.X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        kmeans=cluster.KMeans(n_clusters=params['n_clusters'])
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)
        '''
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        '''
        
        dbscan = cluster.DBSCAN(eps=params['eps'])
        '''
        optics = cluster.OPTICS(min_samples=params['min_samples'],
                                xi=params['xi'],
                                min_cluster_size=params['min_cluster_size'])
        '''
        '''
        affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
        '''
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'])
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')

        clustering_algorithms = (
            ('KMeans',kmeans),
            ('MiniBatchKMeans', two_means),
            #('AffinityPropagation', affinity_propagation),
            ('MeanShift', ms),
            #('SpectralClustering', spectral),
            ('Ward', ward),
            ('AgglomerativeClustering', average_linkage),
            ('DBSCAN', dbscan),
            #('OPTICS', optics),
            ('Birch', birch),
            ('GaussianMixture', gmm)
        )
        finalPred=[]
        finalAlg='KMeans'
        finalPrm=[]
        finalSilht=-1
        #silht_scores=[]
        #accu_scores=[]
        algo_names=[]
        pred_all={}
        for name, algorithm in clustering_algorithms:
            #t0 = time.time()
            algo_names.append(name)
            
            #KMeans tuning number of clusters
            if name=='KMeans':
                print("current algorithm is",name)
                clustNum=[2,3,4,5,6,7,8]
                print("n_clusters: ",clustNum)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                for cnt, clst in enumerate(clustNum):
                    algorithm=cluster.KMeans(n_clusters=clst)
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt, clst,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,clustNum, estClust, finalPred, redY)
            
            #MiniBatchKMeans tuning number of clusters
            if name=='MiniBatchKMeans':
                print("current algorithm is",name)
                clustNum=[2,3,4,5,6,7,8]
                print("n_clusters: ",clustNum)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                for cnt, clst in enumerate(clustNum):
                    algorithm=cluster.MiniBatchKMeans(n_clusters=clst)
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt, clst,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,clustNum, estClust, finalPred, redY)
            ################
            
            elif name=='MeanShift':
                print("current algorithm is",name)
                curQuant=[.1,.15,.2,.25,.3,.4,.5]
                print("Quantile: : ",curQuant)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                for cnt, quant in enumerate(curQuant):
                    bandwidth = cluster.estimate_bandwidth(self.X, quantile=quant)
                    algorithm=cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt,quant,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,curQuant,estClust, finalPred, redY)

            
            elif name=='Ward':
                print("current algorithm is",name)
                curParams=[{'n_clusters': 2, 'connectivity': connectivity},
                {'n_clusters': 3, 'connectivity': connectivity},
                {'n_clusters': 4, 'connectivity': connectivity},
                {'n_clusters': 5, 'connectivity': connectivity},
                {'n_clusters': 6, 'connectivity': connectivity},
                {'n_clusters': 7, 'connectivity': connectivity},
                {'n_clusters': 8, 'connectivity': connectivity},
                {'n_clusters': 2, 'connectivity': None},
                {'n_clusters': 3, 'connectivity': None},
                {'n_clusters': 4, 'connectivity': None},
                {'n_clusters': 5, 'connectivity': None},
                {'n_clusters': 6, 'connectivity': None},
                {'n_clusters': 7, 'connectivity': None},
                {'n_clusters': 8, 'connectivity': None}
                ]
                print("Parameters: ",curParams)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                paramNum=[list(range(len(curParams)))][0]
                for cnt, curPrm in enumerate(curParams):
                    algorithm=cluster.AgglomerativeClustering(n_clusters=curPrm['n_clusters'], linkage='ward',connectivity=curPrm['connectivity'])
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt,curPrm,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,paramNum, estClust, finalPred, redY)
            
            #Heriachical clustering tuning number of clusters
            elif name=='AgglomerativeClustering':
                print("current algorithm is",name)
                curParams=[{'n_clusters': 2, 'connectivity': connectivity},
                {'n_clusters': 3, 'connectivity': connectivity},
                {'n_clusters': 4, 'connectivity': connectivity},
                {'n_clusters': 5, 'connectivity': connectivity},
                {'n_clusters': 6, 'connectivity': connectivity},
                {'n_clusters': 7, 'connectivity': connectivity},
                {'n_clusters': 8, 'connectivity': connectivity},
                {'n_clusters': 2, 'connectivity': None},
                {'n_clusters': 3, 'connectivity': None},
                {'n_clusters': 4, 'connectivity': None},
                {'n_clusters': 5, 'connectivity': None},
                {'n_clusters': 6, 'connectivity': None},
                {'n_clusters': 7, 'connectivity': None},
                {'n_clusters': 8, 'connectivity': None}
                ]
                print("Parameters: ",curParams)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                paramNum=[list(range(len(curParams)))][0]
                for cnt, curPrm in enumerate(curParams):
                    algorithm=cluster.AgglomerativeClustering(n_clusters=curPrm['n_clusters'],linkage="average", affinity="cityblock",connectivity=curPrm['connectivity'])
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt,curPrm,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,paramNum, estClust, finalPred, redY)
            
            #DBSCAN tuning esp and min_samples
            elif name=='DBSCAN':
                print("current algorithm is",name)
                curParams=[{'eps':0.1, 'min_samples':5},
                {'eps':0.5, 'min_samples':5},
                {'eps':1, 'min_samples':5},
                {'eps':3, 'min_samples':5},
                {'eps':5, 'min_samples':5},
                {'eps':10, 'min_samples':7},
                {'eps':0.1, 'min_samples':7},
                {'eps':0.5, 'min_samples':7},
                {'eps':1, 'min_samples':7},
                {'eps':3, 'min_samples':7},
                {'eps':5, 'min_samples':7},
                {'eps':10, 'min_samples':7},
                {'eps':0.1, 'min_samples':10},
                {'eps':0.5, 'min_samples':10},
                {'eps':1, 'min_samples':10},
                {'eps':3, 'min_samples':10},
                {'eps':5, 'min_samples':10},
                {'eps':10, 'min_samples':10},
                {'eps':0.1, 'min_samples':15},
                {'eps':0.5, 'min_samples':15},
                {'eps':1, 'min_samples':15},
                {'eps':3, 'min_samples':15},
                {'eps':5, 'min_samples':15},
                {'eps':10, 'min_samples':15}
                ]
                print("Parameters: ",curParams)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                paramNum=[list(range(len(curParams)))][0]
                for cnt, curPrm in enumerate(curParams):
                    algorithm=cluster.DBSCAN(eps=curPrm['eps'], min_samples=curPrm['min_samples'])
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt,curPrm,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,paramNum, estClust, finalPred, redY)

            
            #################
            #Birch tuning number of clusters  
            
            elif name=='Birch':
                print("current algorithm is",name)
                clustNum=[2,3,4,5,6,7,8,9]
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                print("n_clusters: ",clustNum)
                for cnt, clst in enumerate(clustNum):
                    algorithm=cluster.Birch(n_clusters=clst)
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt,clst,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,clustNum, estClust, finalPred, redY)
            
            elif name=='GaussianMixture':
                print("current algorithm is",name)
                curParams=[{'n_components':2, 'covariance_type':'full'},
                {'n_components':2, 'covariance_type':'tied'},
                {'n_components':2, 'covariance_type':'diag'},
                {'n_components':2, 'covariance_type':'spherical'},
                {'n_components':3, 'covariance_type':'full'},
                {'n_components':3, 'covariance_type':'tied'},
                {'n_components':3, 'covariance_type':'diag'},
                {'n_components':3, 'covariance_type':'spherical'},
                {'n_components':4, 'covariance_type':'full'},
                {'n_components':4, 'covariance_type':'tied'},
                {'n_components':4, 'covariance_type':'diag'},
                {'n_components':4, 'covariance_type':'spherical'},
                {'n_components':5, 'covariance_type':'full'},
                {'n_components':5, 'covariance_type':'tied'},
                {'n_components':5, 'covariance_type':'diag'},
                {'n_components':5, 'covariance_type':'spherical'},
                {'n_components':6, 'covariance_type':'full'},
                {'n_components':6, 'covariance_type':'tied'},
                {'n_components':6, 'covariance_type':'diag'},
                {'n_components':6, 'covariance_type':'spherical'}

                ]
                print("Parameters: ",curParams)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                paramNum=[list(range(len(curParams)))][0]
                for cnt, curPrm in enumerate(curParams):
                    algorithm=mixture.GaussianMixture(n_components=curPrm['n_components'], covariance_type=curPrm['covariance_type'])
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt,curPrm,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,paramNum, estClust, finalPred, redY)
            allPredDF=pd.DataFrame.from_dict(pred_all)
            allPredDF.to_csv(self.outpath+'_allPred.csv',index=False)
        return finalPred, finalAlg, finalPrm, finalSilht, allPredDF

    
    
    
            
''''
            elif name=='OPTICS':
                print("current algorithm is",name)
                curParams=[{'eps':0.1, 'min_samples':5},
                {'eps':0.5, 'min_samples':5},
                {'eps':1, 'min_samples':5},
                {'eps':3, 'min_samples':5},
                {'eps':5, 'min_samples':5},
                {'eps':10, 'min_samples':7},
                {'eps':0.1, 'min_samples':7},
                {'eps':0.5, 'min_samples':7},
                {'eps':1, 'min_samples':7},
                {'eps':3, 'min_samples':7},
                {'eps':5, 'min_samples':7},
                {'eps':10, 'min_samples':7},
                {'eps':0.1, 'min_samples':10},
                {'eps':0.5, 'min_samples':10},
                {'eps':1, 'min_samples':10},
                {'eps':3, 'min_samples':10},
                {'eps':5, 'min_samples':10},
                {'eps':10, 'min_samples':10},
                {'eps':0.1, 'min_samples':15},
                {'eps':0.5, 'min_samples':15},
                {'eps':1, 'min_samples':15},
                {'eps':3, 'min_samples':15},
                {'eps':5, 'min_samples':15},
                {'eps':10, 'min_samples':15}
                ]
                print("Parameters: ",curParams)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                paramNum=[list(range(len(curParams)))][0]
                for cnt, curPrm in enumerate(curParams):
                    algorithm=cluster.OPTICS(eps=curPrm['eps'], min_samples=curPrm['min_samples'])
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt,curPrm,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,paramNum, estClust, umapY, finalPred)
                
            elif name=='AffinityPropagation':
                print("current algorithm is",name)
                apPred={}
                curParams=[{'damping': .5, 'preference': -220},
                {'damping': .6, 'preference': -220},
                {'damping': .75, 'preference': -220},
                {'damping': .9, 'preference': -220},
                {'damping': .5, 'preference': -210},
                {'damping': .6, 'preference': -210},
                {'damping': .75, 'preference': -210},
                {'damping': .9, 'preference': -210},
                {'damping': .5, 'preference': -200},
                {'damping': .6, 'preference': -200},
                {'damping': .75, 'preference': -200},
                {'damping': .9, 'preference': -200},
                {'damping': .5, 'preference': None},
                {'damping': .6, 'preference': None},
                {'damping': .75, 'preference': None},
                {'damping': .9, 'preference': None},
                ]
                print("Parameters: ",curParams)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                paramNum=[list(range(len(curParams)))][0]
                for cnt, curPrm in enumerate(curParams):
                    algorithm=cluster.AffinityPropagation(damping=curPrm['damping'], preference=curPrm['preference'])
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt, curPrm, name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,paramNum, estClust, umapY, finalPred)
                
                
            elif name=='SpectralClustering':
                print("current algorithm is",name)
                clustNum=[2,3,4,5,6,7,8]
                print("n_clusters: ",clustNum)
                silht_scores=[]
                accu_scores=[]
                estClust=[]
                for cnt, clst in enumerate(clustNum):
                    algorithm=cluster.SpectralClustering(n_clusters=clst)
                    finalPred, finalAlg, finalPrm, finalSilht=self.modelFit(cnt,clst,name, algorithm, self.label, self.X, silht_scores, accu_scores,estClust, finalPred, finalAlg, finalPrm, finalSilht,pred_all)
                self.pltScores (name, silht_scores, accu_scores,clustNum, estClust, finalPred, umapY)
'''