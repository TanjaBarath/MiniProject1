import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from KNN import KNN

def hepatitis_data():
    hep_names = ["Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise", "Anorexia",
                 "Liver_Big", "Liver_Firm", "Spleen_Palpable", "Spiders", "Ascites", "Varices",
                 "Bilirubin", "Alk_Phosphate", "Sgot", "Albumin", "Protime", "Histology"]
    hep_df = pd.read_csv('hepatitis.csv', header=None, names=hep_names)

    # eliminating rows with ? values in the dataframe
    hep_df.drop(hep_df.index[hep_df.eq('?').any(1)], inplace=True)

    # notice the ? made some columns non-numeric
    hep_df = hep_df.apply(pd.to_numeric, errors='ignore')

    # hepatitis data has Class 1 and 2, change this to 0 and 1
    hep_df.loc[:, 'Class'] = hep_df['Class'].apply(lambda x: x-1)

    return hep_df

def hep_stats(dataset):
    # first 5 rows of the data and description
    print(dataset.head().to_string(), "\n")
    print(dataset.describe(include='all').to_string())

    # useful statistics
    # total number of rows after dropping
    num_rows = dataset.shape[0]
    print("\nNumber of rows/data points in the dataset after dropping: ", num_rows)

    # number of people that have died and lived
    group = dataset.groupby('Class')
    print("The number of people in the dataset that have died: ", group.size()[1])
    print("The number of people in the dataset that have lived: ", group.size()[2], "\n")

    # distribution of Age versus Class
    dataset.boxplot(column='Age', by='Class')
    plt.title('Distribution of Age for every Class')
    plt.suptitle('')
    plt.ylabel('Age')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    # plt.show()

    # Male and Female counts for each Class
    sex_dist = dataset.groupby(['Class', 'Sex']).size().unstack(fill_value=0).stack().reset_index(name='count')
    x = np.arange(2)
    width = 0.2
    plt.bar(x-0.1, sex_dist.loc[sex_dist['Sex']==1]['count'].to_numpy(), width, color='cyan')
    plt.bar(x+0.1, sex_dist.loc[sex_dist['Sex']==2]['count'].to_numpy(), width, color='green')
    plt.title('Count of Sex for each Class')
    plt.xticks(x, ['Die', 'Live'])
    plt.legend(['Male', 'Female'])
    # plt.show()

    # count of yes/no features based on class
    Steroid = dataset.groupby(['Class', 'Steroid']).size().reset_index(name='count')
    Antivirals = dataset.groupby(['Class', 'Antivirals']).size().reset_index(name='count')
    Fatigue = dataset.groupby(['Class', 'Fatigue']).size().reset_index(name='count')
    Malaise = dataset.groupby(['Class', 'Malaise']).size().reset_index(name='count')
    Anorexia = dataset.groupby(['Class', 'Anorexia']).size().unstack(fill_value=0).stack().reset_index(name='count')
    LiverBig = dataset.groupby(['Class', 'Liver_Big']).size().unstack(fill_value=0).stack().reset_index(name='count')
    LiverFirm = dataset.groupby(['Class', 'Liver_Firm']).size().reset_index(name='count')
    SpleenPalpable = dataset.groupby(['Class', 'Spleen_Palpable']).size().reset_index(name='count')
    Spiders = dataset.groupby(['Class', 'Spiders']).size().reset_index(name='count')
    Ascites = dataset.groupby(['Class', 'Ascites']).size().reset_index(name='count')
    Varices = dataset.groupby(['Class', 'Varices']).size().reset_index(name='count')
    Histology = dataset.groupby(['Class', 'Histology']).size().reset_index(name='count')

    # add all the above into a table with Class, Answer and Feature Counts
    count_dist = pd.DataFrame({'Class': [0, 0, 1, 1], 'Answer': [1, 2, 1, 2], 'Steroid': Steroid['count'],
                               'Antivirals': Antivirals['count'], 'Fatigue': Fatigue['count'], 'Malaise': Malaise['count'],
                               'Anorexia': Anorexia['count'], 'Liver_Big': LiverBig['count'], 'Liver_Firm': LiverFirm['count'],
                               'Spleen_Palpable': SpleenPalpable['count'], 'Spiders': Spiders['count'], 'Ascites': Ascites['count'],
                               'Varices': Varices['count'], 'Histology': Histology['count']})

    # making two plots for yes/no features, first one only has people who died (Class == 0)
    x = np.arange(12)
    width = 0.2
    plt.bar(x-0.1, count_dist.loc[(count_dist['Class'] == 0) & (count_dist['Answer'] == 1)].to_numpy().flatten()[2:],
            width, color='cyan')
    plt.bar(x+0.1, count_dist.loc[(count_dist['Class'] == 0) & (count_dist['Answer'] == 2)].to_numpy().flatten()[2:],
            width, color='green')
    plt.title('Count of Yes/No Answer for Different Features in Class Die')
    plt.xticks(x, ['Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'Liver_Big', 'Liver_Firm',
                               'Spleen_Palpable', 'Spiders', 'Ascites', 'Varices', 'Histology'], rotation=90)
    plt.ylabel('Count')
    plt.legend(['No', 'Yes'])
    # plt.show()

    # second graph has people ho have lived (Class == 1)
    plt.bar(x-0.1, count_dist.loc[(count_dist['Class'] == 1) & (count_dist['Answer'] == 1)].to_numpy().flatten()[2:],
            width, color='cyan')
    plt.bar(x+0.1, count_dist.loc[(count_dist['Class'] == 1) & (count_dist['Answer'] == 2)].to_numpy().flatten()[2:],
            width, color='green')
    plt.title('Count of Yes/No Answer for Different Features in Class Live')
    plt.xticks(x, ['Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'Liver_Big', 'Liver_Firm',
                   'Spleen_Palpable', 'Spiders', 'Ascites', 'Varices', 'Histology'], rotation=90)
    plt.ylabel('Count')
    plt.legend(['No', 'Yes'])
    # plt.show()

    # boxplots of the rest of the features vs Class
    # Bilirubin
    dataset.boxplot(column='Bilirubin', by='Class')
    plt.title('Distribution of Bilirubin for every Class')
    plt.suptitle('')
    plt.ylabel('Bilirubin')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    # plt.show()

    # Alk Phosphate
    dataset.boxplot(column='Alk_Phosphate', by='Class')
    plt.title('Distribution of Alk Phosphate for every Class')
    plt.suptitle('')
    plt.ylabel('Alk Phosphate')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    # plt.show()

    # Sgot
    dataset.boxplot(column='Sgot', by='Class')
    plt.title('Distribution of Sgot for every Class')
    plt.suptitle('')
    plt.ylabel('Sgot')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    # plt.show()

    # Albumin
    dataset.boxplot(column='Albumin', by='Class')
    plt.title('Distribution of Albumin for every Class')
    plt.suptitle('')
    plt.ylabel('Albumin')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    # plt.show()

    # Protime
    dataset.boxplot(column='Protime', by='Class')
    plt.title('Distribution of Protime for every Class')
    plt.suptitle('')
    plt.ylabel('Protime')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    # plt.show()

    print(dataset[dataset.columns].corr()['Class'][:].sort_values())

def messidor_data():
    # Not sure about the columns names? The descriptions were not helpful
    messidor_names = ["Quality", "Pre-Screening", "MA_Detection_0.5", "MA_Detection_0.6", "MA_Detection_0.7",
                 "MA_Detection_0.8", "MA_Detection_0.9", "MA_Detection_1.0", "Exudates_0.5",
                 "Exudates_0.6", "Exudates_0.7", "Exudates_0.8", "Exudates_0.9", "Exudates_0.95",
                 "Exudates_0.99", "Exudates_1.0", "Distance", "Diameter", "AM/FM_Classification", "Class"]
    messidor_df = pd.read_csv('messidor_features.csv', header=None, names=messidor_names)

    # the messidor dataset does not have any missing values
    # the dataset does have a feature representing the quality of the image
    # delete 4 rows with bad quality, i.e. Quality==0
    messidor_df.drop(messidor_df.loc[messidor_df["Quality"] == 0])

    # might need more cleaning up based on data values

def KNN_alg(dataset, features, K, L_fold):
    # randomize the data
    random_data = dataset.sample(frac=1)

    # predicting Class (Live or Die) based on important features
    x, y = random_data[features], random_data[['Class']]

    # create train and test data for predicting class based on Age
    (N, D), C = x.shape, y['Class'].max()
    print("instances (N) \t ", N, "\n features (D) \t ", D, "\n classes (C) \t ", C)

    # implementing L-fold cross-validation and getting all errors
    errors = []
    # splitting the dataset into L chunks
    fold_size = int(dataset.shape[0]/L_fold)
    train_fold, test_fold = [], []
    for i in range(0, dataset.shape[0], fold_size):
        train_fold.append(x[i:(i+fold_size)])
        test_fold.append(y[i:(i+fold_size)])
    # the folds are created, iterate through and change test set
    for i in range(L_fold):
        # need to concat the L-1 training chunks for x (feature values) and y (class)
        x_chunk, y_chunk = [], []
        x_chunk.append(train_fold[:i] + train_fold[(i+1):])
        y_chunk.append(test_fold[:i] + test_fold[(i+1):])

        # choose training and test sets
        x_train, y_train = pd.concat(x_chunk[0]), pd.concat(y_chunk[0])
        x_test, y_test = train_fold[i], test_fold[i]

        # fitting the model
        model = KNN(K=K)
        model.fit(x_train, y_train)
        y_prob, knns = model.predict(x_test)

        # choose class that has the max probability
        # adding one since Class has 1 and 2 as values
        y_pred = np.argmax(y_prob, axis=-1)
        # y_test is a dataframe, convert it to a numpy array and calculate accuracy
        acc = model.evaluate_acc(y_pred, y_test['Class'].to_numpy())
        errors.append(acc)

    print("\nKNN for K = ", K, " on given dataset")
    print("Errors from cross validation: ", errors)
    print("Mean of errors: ", np.mean(errors))
    print("Variance of errors: ", np.var(errors), "\n")

def main():
    np.random.seed(123456)
    data = hepatitis_data()
    #hep_stats(data)
    features = ['Bilirubin', 'Albumin', 'Protime']
    for K in range(2, 10, 2):
        KNN_alg(data, features, K, L_fold=8)

if __name__ == "__main__":
    main()