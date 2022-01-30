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

    # first 5 rows of the data
    print(hep_df.head().to_string())

    # useful statistics
    # total number of rows after dropping
    num_rows = hep_df.shape[0]
    print("\n Number of rows/data points in the dataset after dropping: ", num_rows)

    # count of each Class type (Die = 1 and Live = 2)
    # group = hep_df.groupby('Class')
    # group.size().plot.bar()
    # plt.show()

    # distribution of Age versus Class
    # hep_df.boxplot(column='Age', by='Class')
    # plt.show()

    # count of yes/no features based on class, could make some plot with these
    # Steroid_count = hep_df.groupby(['Class', 'Steroid']).size().reset_index(name='countSteroid')
    # plt.show()

    # randomize the data
    random_data = hep_df.sample(frac=1)

    # predicting Class (Live or Die) based on Age (for testing)
    x, y = random_data[['Age']], random_data[['Class']]

    # create train and test data for predicting class based on Age
    (N, D), C = x.shape, y['Class'].max()
    print("instances (N) \t ", N, "\n features (D) \t ", D, "\n classes (C) \t ", C)

    # split the dataset into 70 training points and 10 test points
    # reset the indices for convenience, might get rid of it for comparing with original data
    x_train, y_train = x[:70].reset_index(drop=True), y[:70].reset_index(drop=True)
    x_test, y_test = x[70:].reset_index(drop=True), y[70:].reset_index(drop=True)

    # try KNN=5
    model = KNN(K=5)
    model.fit(x_train, y_train)
    y_prob, knns = model.predict(x_test)
    print(x_test)
    print('knns shape:', knns.shape)
    print('y_prob shape:', y_prob.shape)
    print(knns)
    print(y_prob)

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