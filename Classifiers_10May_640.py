import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
from sklearn.neighbors.nearest_centroid import NearestCentroid


def getScore(predictions,trueVals,margin):
    """
    Input=predictions,trueVals,margin

    Output=% of correct predictions based on tolerance margin
    
    This method will get score based on the following
    if predictions is in 1% margin of error range (realValue +/- margin*realValue)
    then it is considered to predict correctly, and score =score+1

    """
    trueVals=np.asarray(trueVals)
    score=0
    length=len(predictions)
    for i in range(length):
        less=trueVals[i] - trueVals[i]*margin
        more=trueVals[i] + trueVals[i]*margin
        if(predictions[i]<=more  and predictions[i]>=less ):
            score=score+1
    percent=score*100/length
    return percent


#path
pathRE=r'D:\DataAnalysis\Sparsh\FinalProj\ratingEnabled.csv'
pathCE=r'D:\DataAnalysis\Sparsh\FinalProj\commentEnabled.csv'

#DataFrame
dfRE=pd.read_csv(pathRE, header=0)
dfCE=pd.read_csv(pathCE, header=0)

#features
features=['category_id', 'negTitle', 'neuTitle', 'posTitle','compdTitle', 'negTg', 'neuTg', 'posTg', 'compdTg', 'title_num', 'PD','PM', 'PY', 'TD', 'TM', 'TY']

#features to be predicted
predict=[ 'likes', 'dislikes','comment_count','views']

choice='y'
while(choice=='y'):
   
    #Data Split in training and validation
    predCol='views'
    train_fullR,test_fullR,y_train_fullR,y_test_fullR=train_test_split(dfRE[features],dfRE[predCol],test_size=0.2, random_state=77)
    train_fullC,test_fullC,y_train_fullC,y_test_fullC=train_test_split(dfCE[features],dfCE[predCol],test_size=0.2, random_state=77)

    trainR=train_fullR[0:2000]
    testR=test_fullR[0:2000]
    y_trainR=y_train_fullR[0:2000]
    y_testR=y_test_fullR[0:2000]

    trainC=train_fullC[0:2000]
    testC=test_fullC[0:2000]
    y_trainC=y_train_fullC[0:2000]
    y_testC=y_test_fullC[0:2000]

    
    features=['category_id', 'negTitle', 'neuTitle', 'posTitle','compdTitle', 'negTg', 'neuTg', 'posTg', 'compdTg', 'title_num', 'PD','PM', 'PY', 'TD', 'TM', 'TY']
    print('What would you like to predict:\n1. Views \n2. Likes \n3. Dislikes \n4. Number of Comments\n')
    ch=int(input('Your choice #:'))
    if(ch==1):
        predCol='views'
##        features.append('likes')
##        features.append('dislikes')
##        features.append('comment_count')
        train=trainR
        test=testR
        y_train=y_trainR
        y_test=y_testR
    elif(ch==2):
        predCol='likes'
        features.append('views')
        features.append('dislikes')
        features.append('comment_count')
        train=trainR
        test=testR
        y_train=y_trainR
        y_test=y_testR
    elif(ch==3):
        predCol='dislikes'
        features.append('likes')
        features.append('views')
        features.append('comment_count')
        train=trainR
        test=testR
        y_train=y_trainR
        y_test=y_testR
    elif(ch==4):
        predCol='comment_count'
        features.append('likes')
        features.append('dislikes')
        features.append('views')
        train=trainC
        test=testC
        y_train=y_trainC
        y_test=y_testC
    else:
        print('Out of Choices')
        exit()
    
    

    per=np.zeros((4,19))

    i=0.00
    cnt=0
    while i<=0.95:
        #Naive Bayes
        """
        This will predict likes based on various features 
        """
        i=i+0.05
        print('For tolerance value :',i)
        gnb = GaussianNB()
        gnb.fit(train,y_train)
        ypredict=gnb.predict(test)
        z=y_test
        percent=getScore(ypredict,z,i)
        print('Percent Gauss Naive Bayes:',percent)
        per[0][cnt]=percent
        #Decision Tree
        """
        """
        ##dt=tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
        ##dt.fit(dfRE[features],dfRE['likes'])
        ##ypred=dt.predict(test)
        ##percent=getScore(ypred,y_test['likes'],0.05)
        ##print('Percent:',percent)
        #k means clustering
        ##"""
        ##"""
    ##    kmeans = KMeans(n_clusters=10, random_state=77)
    ##    kmeans.fit(dfRE[features],dfRE['likes'])
    ##    ypred=kmeans.predict(test)
    ##    percent=getScore(ypred,y_test['likes'],i)
    ##    print('Percent K means:',percent)
    ##    per[1][cnt]=percent
        ###ANN
        ##"""
        ##"""
        ##
        ###Support Vector Machine
        ##"""
        ##"""
    ##    sv=svm.SVR(gamma='auto')
    ##    sv.fit(dfRE[features],dfRE['likes'])
    ##    ypred=sv.predict(test)
    ##    percent=getScore(ypred,y_test['likes'],i)
    ##    print('Percent SVM:',percent)
        #Linear Regression
        """
        """
        reg=LinearRegression()
        reg.fit(train,y_train)
        ypred=reg.predict(test)
        percent=getScore(ypred,y_test,i)
        print('Percent Linear Regress:',percent)
        per[1][cnt]=percent
        #Random Forest
        """
        """
        ##rf=RandomForestClassifier(n_estimators=100, max_depth=2,random_state=70)
        ##rf.fit(dfRE[features],dfRE['likes'])
        ##ypred=rf.predict(test)
        ##percent=getScore(ypred,y_test['likes'],0.1)
        ##print('Percent Random Forest:',percent)
        #Nearest Neighbors
        """
        """
        nn=KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
        nn.fit(train,y_train)
        ypred=nn.predict(test)
        percent=getScore(ypred,y_test,i)
        print('Percent Nearest Neighbors:',percent)
        per[2][cnt]=percent
        #apriori
        """
        """
        #nearest Centroid
        """
        """
        nc=NearestCentroid()
        nc.fit(train,y_train)
        ypred=nc.predict(test)
        percent=getScore(ypred,y_test,i)
        print('Percent Nearest Centroid:',percent)
        per[3][cnt]=percent
        cnt=cnt+1
    steps=np.zeros((19,1))
    z=0.05
    for i in range(19):
        steps[i]=z
        z=z+0.05
    plt.plot(steps,per[0],'r--', steps,per[1],'b--', steps,per[2],'k--', steps,per[3],'g--')
    plt.xlabel('Tolerance')
    plt.ylabel('Accuracy')
    plt.title('Linear Regression & Decision Tree Accuracy by Tolerance')
    plt.legend(['GuassianNB', 'Linear Regress','KNeighbors','NearestCentroid'])
    plt.show()    
    choice=input('\nDo you want to continue?(y/n):')


