import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from KNN import KNN

def hepatitis_data():
    hep_names = ["Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise", "Anorexia",
                 "Liver_Big", "Liver_Firm", "Spleen_Palpable", "Spiders", "Ascites", "Varices",
                 "Bilirubin", "Alk_Phosphate", "Sgot", "Albumin", "Protime", "Histology"]
    hep_df = pd.read_csv('hepatitis.csv', header=None, names=hep_names)

    # eliminating rows with ? values in the dataframe
    hep_df.drop(hep_df.index[hep_df.eq('?').any(1)], inplace=True)

    # first 5 rows of the data and description
    print(hep_df.head().to_string())
    print(hep_df.describe(include='all').to_string())
    # notice the ? made some columns non-numeric
    hep_df = hep_df.apply(pd.to_numeric, errors='ignore')
    print(hep_df.describe(include='all').to_string())

    # useful statistics
    # total number of rows after dropping
    num_rows = hep_df.shape[0]
    print("\n Number of rows/data points in the dataset after dropping: ", num_rows, "\n")

    # count of each Class type (Die = 1 and Live = 2)
    group = hep_df.groupby('Class')
    group.size().plot.bar()
    plt.title('Count of each Class')
    plt.xlabel('')
    plt.ylabel('Count')
    plt.xticks(np.arange(2), ['Die', 'Live'])
    plt.show()

    # distribution of Age versus Class
    hep_df.boxplot(column='Age', by='Class')
    plt.title('Distribution of Age for every Class')
    plt.suptitle('')
    plt.ylabel('Age')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    plt.show()

    # Male and Female counts for each Class
    sex_dist = hep_df.groupby(['Class', 'Sex']).size().unstack(fill_value=0).stack().reset_index(name='count')
    x = np.arange(2)
    width = 0.2
    plt.bar(x-0.1, sex_dist.loc[sex_dist['Sex']==1]['count'].to_numpy(), width, color='cyan')
    plt.bar(x+0.1, sex_dist.loc[sex_dist['Sex']==2]['count'].to_numpy(), width, color='green')
    plt.title('Count of Sex for each Class')
    plt.xticks(x, ['Die', 'Live'])
    plt.legend(['Male', 'Female'])
    plt.show()

    # count of yes/no features based on class
    Steroid = hep_df.groupby(['Class', 'Steroid']).size().reset_index(name='count')
    Antivirals = hep_df.groupby(['Class', 'Antivirals']).size().reset_index(name='count')
    Fatigue = hep_df.groupby(['Class', 'Fatigue']).size().reset_index(name='count')
    Malaise = hep_df.groupby(['Class', 'Malaise']).size().reset_index(name='count')
    Anorexia = hep_df.groupby(['Class', 'Anorexia']).size().unstack(fill_value=0).stack().reset_index(name='count')
    LiverBig = hep_df.groupby(['Class', 'Liver_Big']).size().unstack(fill_value=0).stack().reset_index(name='count')
    LiverFirm = hep_df.groupby(['Class', 'Liver_Firm']).size().reset_index(name='count')
    SpleenPalpable = hep_df.groupby(['Class', 'Spleen_Palpable']).size().reset_index(name='count')
    Spiders = hep_df.groupby(['Class', 'Spiders']).size().reset_index(name='count')
    Ascites = hep_df.groupby(['Class', 'Ascites']).size().reset_index(name='count')
    Varices = hep_df.groupby(['Class', 'Varices']).size().reset_index(name='count')
    Histology = hep_df.groupby(['Class', 'Histology']).size().reset_index(name='count')

    # add all the above into a table with Class, Answer and Feature Counts
    count_dist = pd.DataFrame({'Class': [1, 1, 2, 2], 'Answer': [1, 2, 1, 2], 'Steroid': Steroid['count'],
                               'Antivirals': Antivirals['count'], 'Fatigue': Fatigue['count'], 'Malaise': Malaise['count'],
                               'Anorexia': Anorexia['count'], 'Liver_Big': LiverBig['count'], 'Liver_Firm': LiverFirm['count'],
                               'Spleen_Palpable': SpleenPalpable['count'], 'Spiders': Spiders['count'], 'Ascites': Ascites['count'],
                               'Varices': Varices['count'], 'Histology': Histology['count']})

    # making two plots for yes/no features, first one only has people who died (Class == 1)
    x = np.arange(12)
    width = 0.2
    plt.bar(x-0.1, count_dist.loc[(count_dist['Class'] == 1) & (count_dist['Answer'] == 1)].to_numpy().flatten()[2:],
            width, color='cyan')
    plt.bar(x+0.1, count_dist.loc[(count_dist['Class'] == 1) & (count_dist['Answer'] == 2)].to_numpy().flatten()[2:],
            width, color='green')
    plt.title('Count of Yes/No Answer for Different Features in Class Die')
    plt.xticks(x, ['Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'Liver_Big', 'Liver_Firm',
                               'Spleen_Palpable', 'Spiders', 'Ascites', 'Varices', 'Histology'], rotation=90)
    plt.ylabel('Count')
    plt.legend(['No', 'Yes'])
    plt.show()

    # second graph has people ho have lived (Class == 2)
    plt.bar(x-0.1, count_dist.loc[(count_dist['Class'] == 2) & (count_dist['Answer'] == 1)].to_numpy().flatten()[2:],
            width, color='cyan')
    plt.bar(x+0.1, count_dist.loc[(count_dist['Class'] == 2) & (count_dist['Answer'] == 2)].to_numpy().flatten()[2:],
            width, color='green')
    plt.title('Count of Yes/No Answer for Different Features in Class Live')
    plt.xticks(x, ['Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'Liver_Big', 'Liver_Firm',
                   'Spleen_Palpable', 'Spiders', 'Ascites', 'Varices', 'Histology'], rotation=90)
    plt.ylabel('Count')
    plt.legend(['No', 'Yes'])
    plt.show()

    # boxplots of the rest of the features vs Class
    # Bilirubin
    hep_df.boxplot(column='Bilirubin', by='Class')
    plt.title('Distribution of Bilirubin for every Class')
    plt.suptitle('')
    plt.ylabel('Bilirubin')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    plt.show()

    # Alk Phosphate
    hep_df.boxplot(column='Alk_Phosphate', by='Class')
    plt.title('Distribution of Alk Phosphate for every Class')
    plt.suptitle('')
    plt.ylabel('Alk Phosphate')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    plt.show()

    # Sgot
    hep_df.boxplot(column='Sgot', by='Class')
    plt.title('Distribution of Sgot for every Class')
    plt.suptitle('')
    plt.ylabel('Sgot')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    plt.show()

    # Albumin
    hep_df.boxplot(column='Albumin', by='Class')
    plt.title('Distribution of Albumin for every Class')
    plt.suptitle('')
    plt.ylabel('Albumin')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    plt.show()

    # Protime
    hep_df.boxplot(column='Protime', by='Class')
    plt.title('Distribution of Protime for every Class')
    plt.suptitle('')
    plt.ylabel('Protime')
    plt.xlabel('')
    plt.xticks(np.arange(4), ['', 'Die', 'Live', ''])
    plt.show()

    # randomize the data
    random_data = hep_df.sample(frac=1)

    # predicting Class (Live or Die) based on Age (for testing)
    x, y = random_data[['Age']], random_data[['Class']]

    # create train and test data for predicting class based on Age
    (N, D), C = x.shape, y['Class'].max()
    print("instances (N) \t ", N, "\n features (D) \t ", D, "\n classes (C) \t ", C)

    # split the dataset into 70 training points and 10 test points
    x_train, y_train = x[:70], y[:70]
    x_test, y_test = x[70:], y[70:]

    # try KNN=5
    model = KNN(K=5)
    model.fit(x_train, y_train)
    y_prob, knns = model.predict(x_test)
    print('knns shape:', knns.shape)
    print('y_prob shape:', y_prob.shape)

    # choose class that has the max probability
    # adding one since Class has 1 and 2 as values
    y_pred = np.argmax(y_prob, axis=-1) + 1
    # y_test is a dataframe, convert it to a numpy array and calculate accuracy
    acc = model.evaluate_acc(y_pred, y_test['Class'].to_numpy())
    print(acc)

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

def main():
    np.random.seed(123456)
    hepatitis_data()

if __name__ == "__main__":
    main()