import numpy as np
import pandas as pd

def KNN():
    print("...")

def Decision_Tree():
    print("...")

def hepatitis_data():
    hep_names = ["Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise", "Anorexia",
                 "Liver_Big", "Liver_Firm", "Spleen_Palpable", "Spiders", "Ascites", "Varices",
                 "Bilirubin", "Alk_Phosphate", "Sgot", "Albumin", "Protime", "Histology"]
    hep_df = pd.read_csv('hepatitis.csv', header=None, names=hep_names)

    # eliminating rows with ? values in the dataframe
    hep_df.drop(hep_df.index[hep_df.eq('?').any(1)], inplace=True)

    print(hep_df.to_string())

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

    print(messidor_df.to_string())

def main():
    messidor_data()

if __name__ == "__main__":
    main()