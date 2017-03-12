__author__ = 'Juan Harrington' # please type your name

import pandas as pd
import numpy as np
# import the functions for cosine distance, euclidean distance
from scipy.spatial.distance import cosine, euclidean, correlation

# define a functions, which takes the given item as the input and returns the top K similar items (in a data frame)
def top_k_items(dense_matrix, item_number, k):
    # copy the dense matrix and transpose it so each row represents an item
    df_sim = dense_matrix.transpose()
    # remove the active item 
    df_sim = df_sim.loc[df_sim.index != item_number]
    # calculate the distance between the given item for each row (apply the function to each row if axis = 1)
    df_sim["distance"] = df_sim.apply(lambda x: euclidean(dense_matrix[item_number], x), axis=1)
    # return the top k from the sorted distances
    return df_sim.sort_values(by="distance").head(k)["distance"]   

# define a functions, which takes the given user as the input and returns the top K similar users (in a data frame)
def top_k_users(dense_matrix, user_number, k):
    # no need to transpose the matrix this time because the rows already represent users
    # remove the active user
    df_sim = dense_matrix.loc[dense_matrix.index != user_number]
    # calculate the distance for between the given user and each row
    df_sim["distance"] = df_sim.apply(lambda x: euclidean(dense_matrix.loc[user_number], x), axis=1)
    # return the top k from the sorted distances
    return df_sim.sort_values(by="distance").head(k)["distance"] 

# please note that I have changed this function in the ml_100k.py a little. Now this function has two
# additional arguments: k (i.e., the number of similar user) with the default value 5, and item_number that 
# represent the number of the item. You want to predict the rating the user will give to the item. 
def user_based_predict(df_train_x, df_test_x, df_train_y, user_number, item_number, k=5):    
    # copy from all the training predictors
    df_sim = df_train_x.copy()
    # for each user, calculate the distance between this user and the active user
    df_sim["distance"] = df_sim.apply(lambda x: euclidean(df_test_x.loc[user_number], x), axis=1)
    # create a new data frame to store the top k similar users
    df_sim_users = df_sim.loc[df_sim.sort_values(by="distance").head(k).index]    
    # calculate these similar users' rating on a given item, weighted by distance
    df_sim_users["weighed_d"] = map(lambda x: df_sim_users.loc[x]["distance"]*df_train_y.loc[x][item_number], df_sim_users.index)
    predicted = df_sim_users["weighed_d"].sum()/df_sim_users["distance"].sum()
   
    return predicted

def main():
    # read the file
    df = pd.read_csv("C:\\Users\\jharrington\\Documents\\_DSU-MSA\\INFS770\\Assignment3\\DBbook_train_ratings.tsv", # the location to the data file
                       sep="\t" # for tab delimited documents, use "\t" as the seperator
                       )
    #print df.head()

    #Q1. How many unique users and books are there in this data set
    print "Q1:"
    print "Ratings: %d" % len(df)
    print "Unique users: %d" % len(df["userID"].unique())
    print "Unique items: %d" % len(df["itemID"].unique())

    # create a pivot table
    dense_matrix = df.pivot_table(values="rate", index=["userID"], columns=["itemID"], aggfunc=np.sum)
    print "Shape of the user-item matrix: %d x %d" % dense_matrix.shape
    #print dense_matrix.head()

    #Q2. How many cells in the utility matrix are not populated? What is the percentage?
    print "Q2:"
    print dense_matrix.isnull().sum().sum() 
    #38036488/38112046 99.8017%
    
    #replace missing values (NaN) with 0's to calculate correlation
    dense_matrix = dense_matrix.fillna(0)

    #Q3. What is the correlation distance between Users 2 and 3? Are they similar?
    print "Q3:"
    print "Between users 2 and 3: ", correlation(dense_matrix.iloc[1], dense_matrix.iloc[2])

    #Q4. Between books 18 and 36, which is more similar to Book 1?
    print "Q4:"
    print "Between 18 and 36: ", correlation(dense_matrix[18], dense_matrix[36])
    print "Between 1 and 18: ", correlation(dense_matrix[1], dense_matrix[18])
    print "Between 1 and 36: ", correlation(dense_matrix[1], dense_matrix[36])

    #Q5. Which 5 books are most similar to Book 8010
    print "Q5:"
    print top_k_items(dense_matrix, 8010, 5)
   
    #Q6. Remove books and users with less than 20 rating scores
    df_item_freq = df.groupby("itemID").count()
    df_user_freq = df.groupby("userID").count()
    selected_items = df_item_freq[df_item_freq["userID"]>20].index
    dense_matrix = dense_matrix[selected_items]
    selected_users = df_user_freq[df_user_freq["itemID"]>20].index
    dense_matrix = dense_matrix.loc[selected_users]

    #Q7. Partition the data set for cross validating the performance on predicting rating on Book 8010. 
    print "Q7:"
    dense_matrix[8010].value_counts()

    # create a data frame for the predictors
    df_x = dense_matrix[[col for col in dense_matrix.columns if col != 8010]]
    print df_x.shape

    # create a series for the outcome
    df_y = dense_matrix[[8010]]
    print df_y.shape

    # partion for cross validation
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=0)
    df_train_x = pd.DataFrame(train_x, columns=df_x.columns)
    df_test_x = pd.DataFrame(test_x, columns=df_x.columns)
    df_train_y = pd.DataFrame(train_y, columns=[8010])
    df_test_y = pd.DataFrame(test_y, columns=[8010])
    print "shapes"
    print "train x:", df_train_x.shape
    print "test x:", df_test_x.shape
    print "train y:", df_train_y.shape
    print "test y:", df_test_y.shape
    print 
    print "class counts"
    print df_train_y[8010].value_counts()
    print df_test_y[8010].value_counts()
    print "means"
    #print df_x[8010].mean() 
    #print df_y[8010].mean() 

    #Q8
    print "Q8:"
    user_number = df_test_x.index[23]
    print "This user's rating on other items: ", df_test_x.loc[user_number]
    print
    print "Predicted rating on Item 8010:", user_based_predict(df_train_x, df_test_x, df_train_y, user_number, 8010, k=10)
    print "True rating on Item 8010:     ", df_test_y.loc[user_number][8010]

    #Q9
    print "Q9:"
    pred_8010 = []
    for user_number in df_test_x.index:
        predicted = user_based_predict(df_train_x, df_test_x, df_train_y, user_number, 8010, k=10)
        pred_8010.append(predicted)

    from sklearn.metrics import mean_absolute_error
    print "Mean Absolute Error: ", mean_absolute_error(pred_8010, df_test_y[8010])

if __name__ == "__main__":
    main()