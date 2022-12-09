import pandas as pd
from sqlalchemy import create_engine
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from DecisionTree.decisionTree import *
from Percetron.percetron import *

def connectDb():
    # Create an engine instance
    alchemyEngine = create_engine('postgresql+psycopg2://postgres:123@localhost:5432/iot', pool_recycle=3600);
    # Connect to PostgreSQL server
    dbConnection = alchemyEngine.connect();
    # Read data from PostgreSQL database table and load into a DataFrame instance
    df  = pd.read_sql("select * from \"HR_comma_sep\"", dbConnection);
    return (df , dbConnection)
def cleanData(df):
    df = df.rename(columns={'satisfaction_level': 'satisfaction',
                            'last_evaluation': 'evaluation',
                            'number_project': 'projectCount',
                            'average_montly_hours': 'averageMonthlyHours',
                            'time_spend_company': 'yearsAtCompany',
                            'Work_accident': 'workAccident',
                            'promotion_last_5years': 'promotion',
                            'sales': 'department',
                            'left': 'turnover'
                            })

    back = df['turnover']
    df.drop(labels=['turnover'], axis=1, inplace=True)
    df.insert(9, 'turnover', back)
    return df
def splitData(data):
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.2, random_state=41)
    return (X_train, X_test , Y_train , Y_test) ;

def trainModel(algorithm , df , dbConnection):
    accurancy = 0
    f1 = 0
    recall = 0
    precision = 0
    encoder = ce.OrdinalEncoder(cols=['department', 'salary'])
    data = encoder.fit_transform(df)
    X_train, X_test, Y_train, Y_test = splitData(data)

    if algorithm == 1:
        classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
        classifier.fit(X_train,Y_train)
        # classifier.print_tree()
        accurancy = accuracy_score(Y_test , classifier.predict(X_test))
        f1 = f1_score(Y_test , classifier.predict(X_test))
        recall = recall_score(Y_test , classifier.predict(X_test))
        precision = precision_score(Y_test , classifier.predict(X_test))
    elif algorithm == 2:
        percep = Perceptron(learning_rate = 0.01 , n_iters = 1000)
        percep.fit(X_train, Y_train)
        accurancy = accuracy_score(Y_test, percep.predict(X_test))
        f1 = f1_score(Y_test, percep.predict(X_test))
        recall = recall_score(Y_test, percep.predict(X_test))
        precision = precision_score(Y_test, percep.predict(X_test))

    sqlCheck = "SELECT * FROM \"result\" WHERE algor = " + str(algorithm)
    sqlRes =  "INSERT INTO \"result\" values (" + str(accurancy) +","+ str(f1) + "," + str(recall) + "," + str(precision) +"," + str(algorithm) + ")"
    sqlUpdate = "UPDATE \"result\" SET accurancy = " + str(accurancy) +", f1_score = " +str(f1) + ", recall = " + str(recall) + ", precision = " + str(precision) + "where algor = " + str(algorithm)
    row = dbConnection.execute(sqlCheck)
    if row == None:
        dbConnection.execute(sqlRes)
    else:
        dbConnection.execute(sqlUpdate)




# corr = df.corr()
# corr = (corr)
# sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# print(corr)
# plt.show()



def main():
    init = connectDb()
    df = init[0]
    dbConnection = init[1]
    df = cleanData(df)
    trainModel(2 , df , dbConnection)

if __name__ == "__main__":
    main()