import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn.categorical import barplot
from sklearn import preprocessing
import numpy as np

class CleanTheMess:
    def __init__(self, dt=None, path='./', label=0):
        self.dt=dt
        self.path=path
        self.label=label
    
    def readFile(self, header=None, sep=','):
        try:
            dt=pd.read_csv(self.path, sep=sep, comment='#',header=header)
        except:
            dt=pd.read_csv(self.path, sep=sep, encoding='cp1252',comment='#',header=header)
        return dt
        
    def dropCols(self,col2rm):
        if not col2rm:
            print("List is empty")
            return
        elif type(col2rm) is str:
            col2rm=[col2rm]
        
        newCol2rm=[]
        for x in col2rm:
            if x in self.dt.columns:
                newCol2rm.append(x)
        dt=self.dt.drop(columns=newCol2rm)
        return dt
    
    def keepCols(self,col2kp):
        if not col2kp:
            print("List is empty")
            return
        elif type(col2kp) is str:
            col2kp=[col2kp]
        
        newCol2kp=[]
        for x in col2kp:
            if x in self.dt.columns:
                newCol2kp.append(x)
        dt=self.dt[newCol2kp]
        return dt
    
    def uniqRows(self, col):
        if not col:
            print("Column list is empty")
            return
        else:
            newCol=[]
            for x in col:
                if x in self.dt.columns:
                    newCol.append(x)
                else:
                    print(x,' is not found in the data table.')
            if not newCol:
                print('No column is found.')
                return
            else:
                dt=self.dt.drop_duplicates(subset=newCol,keep='first')
        return dt
                
    def resetIndex(self):
        tmp=self.dt.reset_index()
        up=tmp.shape[0]
        self.dt=tmp.reindex(index=range(0,up))
        self.dt=self.dt.drop(columns='index')
        return self.dt
    
    def table(self,col2tbl):
        if not col2tbl:
            print("List is empty")
        else:
            newCol2rm=[]
            for x in col2tbl:
                if x in self.dt.columns:
                    newCol2rm.append(x)
            print(self.dt.groupby(col2tbl).size())
    
    def diff(self, list1, list2):
        return (list(list(set(list1)-set(list2)) + list(set(list2)-set(list1))))

    def searchCols(self,term):
        return [col for col in self.dt.columns if term in col]
    
    def quickPlt(self, x, y=None, pltype='bar', rotateX=False, normalization='index'):
        if y==None:
            outname=self.path+x+'_'+pltype+'.png'
        else:
            outname=self.path+x+'_'+y+'_'+pltype+'.png'
        plt.figure(figsize=(12, 8))
        sns.set(font_scale=2)
        if pltype=='bar':
            cntplt=sns.barplot(x=x,y=y, data=self.dt)
            #plt.savefig(outname, bbox_inches = "tight",dpi=300)
        elif pltype=='hist':
            cntplt=sns.histplot(x=x,data=self.dt)
        elif pltype=='scatter':
            cntplt=sns.scatterplot(x=x,y=y,hue=self.label, data=self.dt)
        elif pltype=='box':
            cntplt=sns.boxplot(x=x,y=y, data=self.dt)
        elif pltype=='strip':
            cntplt=sns.stripplot(x=x,y=y, data=self.dt)
        elif pltype=='ratio':
            tb=pd.crosstab(self.dt[x], self.dt[y],normalize=normalization)
            cntplt=sns.heatmap(tb, annot=True, fmt='.2%')
        elif pltype=='freq':
            tb=pd.crosstab(self.dt[x], self.dt[y])
            cntplt=sns.heatmap(tb, annot=True, fmt='g')
        if rotateX==True:
            #cntplt.set_xticklabels(cntplt.get_xticklabels(), rotation=45)
            plt.xticks(rotation=45)
        cntplt.figure.savefig(outname, bbox_inches = "tight",dpi=300)
        return cntplt

    def CheckMissing(self):
        outname=self.path+'_MissingValuesByColumn.png'
        missing = self.dt.isnull().sum()*100/len(self.dt)
        missing_value = pd.DataFrame({'column_name': self.dt.columns,'percent_missing': missing})
        missing_value.sort_values('percent_missing', ascending=False, inplace=True)
        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        #sns.color_palette("rocket")
        sns.set_style("darkgrid",{"axes.facecolor": ".9"})
        sns.set_context("poster")
        barplt=sns.distplot(missing_value["percent_missing"],kde=False,color='darkred')
        barplt.figure.savefig(outname)
        return barplot, missing_value
    
    def DrpConstantCols(self):
        colkeep=[c for c in list(self.dt) if len(self.dt[c].unique())>1]
        dt=self.dt[colkeep]
        return dt
    
    def DrpMissingCols(self, miss_rate=0.6):
        dt=self.dt[self.dt.columns[self.dt.isnull().mean()<miss_rate]]
        return dt
    
    def DrpCorrCols(self, corr=0.95):
        Corr = self.dt.corr().abs()
        upper_tri = Corr.where(np.triu(np.ones(Corr.shape),k=1).astype(np.bool))
        corr_to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr)]
        dt=self.dt.drop(columns=corr_to_drop)
        return dt
    
    def char2num(self):
        tmpobj=self.dt.select_dtypes(include=['object'])
        objcols=tmpobj.columns
        #encode character columns with numbers
        for f in objcols:
            le = preprocessing.LabelEncoder()
            le.fit(list(self.dt[f].values))
            self.dt[f]=le.transform(list(self.dt[f].values))
        return self.dt


def main():
    CleanTheMess()

if __name__ == "__main__":
    main()

