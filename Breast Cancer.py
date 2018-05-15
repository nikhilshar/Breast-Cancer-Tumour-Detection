# -*- coding: utf-8 -*-
"""
Created on Mon May 1 10:13:04 2018

@author: Nikhil Sharma
"""
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import tree

lbc=load_breast_cancer()
clf= tree.DecisionTreeClassifier()

X=lbc.data
Y=lbc.target
#splitting up the DATA in train and test
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=.1)
#We have made a classifier tree
clf=clf.fit(X_train,Y_train)
#Checking out the data Validity over training

print("\n\n\t BREAST CANCER TUMOUR DETECTION ")

print("\nPridiction Accuracy \t")
prediction=clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction))
print("\n\nFrom this Program \n we are predicting Type of tumour\n1. malignant \n2.bengin")
print("\nFeatures of the dataSet (30)")
print(lbc.feature_names)
print("\n\n\n\nTarget Or label \n( Means on which we are mapping data)")
print(lbc.target_names)

choice=str(input("Do you really want to \nEnter 30 values (y/n) "))
if(choice=='y'):  
    mrad=float(input("Enter mean radius:\t"))
    mtex=float(input("Enter mean Texture:\t"))
    mperi=float(input("Enter mean perimeter:\t"))
    marea=float(input("Enter mean area:\t"))
    msmooth=float(input("Enter mean smoothness:\t"))
    mcompact=float(input("Enter mean compactness:\t"))
    mconcave=float(input("Enter mean concavity:\t"))
    mconcavepts=float(input("Enter mean concave pts:\t"))
    msymmetry=float(input("Enter mean Symmetry:\t"))
    mfracdim=float(input("Enter Frac. dimension:\t"))
    rerr=float(input("Enter radius err:\t"))
    texterr=float(input("Enter Text error:\t"))
    perimerr=float(input("Enter perimeter error:\t"))
    areaerr=float(input("Enter area error:\t"))
    smootherr=float(input("Enter smoothness error:\t"))
    compacterr=float(input("Enter compact error:\t"))
    concaveerr=float(input("Enter concave error:\t"))
    concavepts=float(input("Enter Concave pts:\t"))
    symmetryerr=float(input("Enter symmetry error:\t"))
    fracdimerr=float(input("Enter frac dim error:\t"))
    wrad=float(input("Enter worst radius:\t"))
    wtexture=float(input("Enter worst texture:\t"))
    wperi=float(input("Enter worst perimeter:\t"))
    warea=float(input("Enter worst area:\t"))
    wsmooth=float(input("Enter worst smoothness:\t"))
    wcompact=float(input("Enter worst compactness:\t"))
    wconcavity=float(input("Enter worst concavity:\t"))
    wconcavepts=float(input("Enter worst cancave pts:\t"))
    wsymmetry=float(input("Enter worst symmetry:\t"))
    wfracdim=float(input("Enter worst fractal dim:\t"))
    testing_data=[[mrad,mtex,mperi,marea,msmooth,mcompact,mconcave,
              mconcavepts,msymmetry,mfracdim,rerr,texterr,perimerr
              ,areaerr,smootherr,compacterr,concaveerr,concavepts,symmetryerr
              ,fracdimerr,wrad,wtexture,wperi,warea,wsmooth,wcompact
              ,wconcavity,wconcavepts,wsymmetry,wfracdim]]

    new_predict=clf.predict(testing_data)
else:
    
    Ex_test=[[2.057e+01, 1.777e+01, 1.329e+02, 1.326e+03, 8.474e-02, 7.864e-02,
       8.690e-02, 7.017e-02, 1.812e-01, 5.667e-02, 5.435e-01, 7.339e-01,
       3.398e+00, 7.408e+01, 5.225e-03, 1.308e-02, 1.860e-02, 1.340e-02,
       1.389e-02, 3.532e-03, 2.499e+01, 2.341e+01, 1.588e+02, 1.956e+03,
       1.238e-01, 1.866e-01, 2.416e-01, 1.860e-01, 2.750e-01, 8.902e-02]]
    print("Example Data\n",Ex_test)
    new_predict=clf.predict(Ex_test)

print("\nResult:")
if(new_predict[0] == 0):
    print("\nMalignant: Cancer Prone Tumour")
else:
    print("\nBengin : Non Cancer Tumour")
print("\n\nWith Pridiction Accuracy:")
print(100*accuracy_score(Y_test,prediction),"%")

"""
                                           Min    Max
    ===================================== ====== ======
    radius (mean):                        6.981  28.11
    texture (mean):                       9.71   39.28
    perimeter (mean):                     43.79  188.5
    area (mean):                          143.5  2501.0
    smoothness (mean):                    0.053  0.163
    compactness (mean):                   0.019  0.345
    concavity (mean):                     0.0    0.427
    concave points (mean):                0.0    0.201
    symmetry (mean):                      0.106  0.304
    fractal dimension (mean):             0.05   0.097
    radius (standard error):              0.112  2.873
    texture (standard error):             0.36   4.885
    perimeter (standard error):           0.757  21.98
    area (standard error):                6.802  542.2
    smoothness (standard error):          0.002  0.031
    compactness (standard error):         0.002  0.135
    concavity (standard error):           0.0    0.396
    concave points (standard error):      0.0    0.053
    symmetry (standard error):            0.008  0.079
    fractal dimension (standard error):   0.001  0.03
    radius (worst):                       7.93   36.04
    texture (worst):                      12.02  49.54
    perimeter (worst):                    50.41  251.2
    area (worst):                         185.2  4254.0
    smoothness (worst):                   0.071  0.223
    compactness (worst):                  0.027  1.058
    concavity (worst):                    0.0    1.252
    concave points (worst):               0.0    0.291
    symmetry (worst):                     0.156  0.664
    fractal dimension (worst):            0.055  0.208
"""