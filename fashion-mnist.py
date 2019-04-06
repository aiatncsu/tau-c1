import pandas as pd
from sklearn.model_selection import train_test_split

#Data Preparation
def data_prep():
    #Reformatting as given in the problem
    train = pd.read_csv("../input/fashion-mnist_train.csv")
    train_x = train[list(train.columns)[1:]].values
    train_y = train['label'].values

    ## normalize and reshape the predictors  
    train_x = train_x / 255

    ## create train and validation datasets
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

    ## reshape the inputs
    train_x = train_x.reshape(-1, 784)
    val_x = val_x.reshape(-1, 784) 

    return


#Main Function
if __name__ == '__main__':
    data_prep()