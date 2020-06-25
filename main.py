import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def welcome():
    print("Welcome to Predict agent System")
    print("Press ENTER key to proceed")
    input()
def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    for x in content_list:
        if x.split('.')[-1]=='csv':
            csv_files.append(x)
    if len(csv_files)==0:
        return 'No csv file in the directory'
    else:
        return csv_files
def display_and_select_csv(csv_files):
    i=0
    for file_name in csv_files:
        print(i,'...',file_name)
        i+=1
    return csv_files[int(input("Select file to create ML model"))]

def main():
    welcome()
    try:
        csv_files=checkcsv()
        if csv_files=='No csv file in the directory':
            raise FileNotFoundError('No csv file in the directory')
        csv_file=display_and_select_csv(csv_files)
        print(csv_file,'is selected')
        print('Reading csv file')
        print('Creating Dataset')
        dataset=pd.read_csv(csv_file)
        print('Dataset created')
        X=dataset.iloc[:,:-1].values
        Y=dataset.iloc[:,-1].values
        s=float(input("Enter test data size (between 0 and 1)") )
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=s)
        print("Model creation in progression")
        regressionObject=LinearRegression()
        regressionObject.fit(X_train,Y_train)
        print("Model is created")
        print("Press ENTER key to predict test data in trained model")


        # take input for x1
        x1=int(input("Enter value for x1") )

        # take input for x2
        x2=int(input("Enter value for x2") )


        Y_pred=regressionObject.predict([[x1,x2]])

        print(Y_pred)
    except FileNotFoundError:
        print('No csv file in the directory')
        print("Press ENTER key to exit")
        input()
        exit()

if __name__=="__main__":
    main()
    input()

        

