import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    group = hep_df.groupby('Class')
    group.size().plot.bar()
    plt.show()

    # distribution of Age versus Class
    hep_df.boxplot(column='Age', by='Class')
    plt.show()

    # count of yes/no features based on class, could make some plot with these
    # Steroid_count = hep_df.groupby(['Class', 'Steroid']).size().reset_index(name='countSteroid')

    # plt.show()

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