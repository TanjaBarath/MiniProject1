from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from KNN import KNN
from decision_tree import decision_tree


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

    # get correlation matrix
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
    messidor_df = messidor_df.drop(messidor_df.loc[messidor_df["Quality"] == 0].index)

    # might need more cleaning up based on data values

    return messidor_df

def messidor_stats(dataset):
    # first 5 rows of the data and description
    print(dataset.head().to_string(), "\n")
    data = dataset.groupby('Class')
    print(data.describe(include='all').to_string())

    # get correlation matrix
    print(dataset[dataset.columns].corr()['Class'][:].sort_values())

    # seems like all features are not very correlated with Class
    # but here are boxplots for the most correlated features
    # MA Detection 0.5, MA Detection 0.6, MA Detection 0.7, MA Detection 0.8, Exudates 0.99
    df = pd.DataFrame(data = dataset[['Class', 'MA_Detection_0.5', 'MA_Detection_0.6', 'MA_Detection_0.7', 'MA_Detection_0.8', 'Exudates_0.99']],
                      columns = ['Class', 'MA_Detection_0.5', 'MA_Detection_0.6', 'MA_Detection_0.7', 'MA_Detection_0.8', 'Exudates_0.99'])
    df.loc[:, 'Exudates_0.99'] = df['Exudates_0.99'].apply(lambda x: x * 10)
    df = pd.melt(df, id_vars=['Class'], value_vars = ['MA_Detection_0.5', 'MA_Detection_0.6', 'MA_Detection_0.7', 'MA_Detection_0.8', 'Exudates_0.99'], var_name = 'Feature', value_name = 'Value')
    sns.set_theme(style="ticks", palette="pastel")
    plot = sns.boxplot(x='Feature', y='Value', hue='Class', palette=["m","g"], data = df)
    plt.xticks(rotation=45)
    plt.suptitle('')
    plot.legend(title = 'Class')
    legend_labels = ['No DR', 'DR']
    n = 0
    for i in legend_labels:
        plot.legend_.texts[n].set_text(i)
        n += 1
    plt.tight_layout()
    plt.show()

# predicting Class (Live or Die) based on important features using cross validation
def KNN_cross_validation(dataset, train, features, K, L_fold):
    # split the class from the features
    x_train, y_train = train[features], train[['Class']]

    # implementing L-fold cross-validation on train data
    acc = []
    # splitting the training dataset into L chunks
    fold_size = int(train.shape[0]/L_fold)
    x_fold, y_fold = [], []
    for i in range(0, dataset.shape[0], fold_size):
        x_fold.append(x_train[i:(i+fold_size)])
        y_fold.append(y_train[i:(i+fold_size)])
    # the folds are created, iterate through and change validation set
    for i in range(L_fold):
        # need to concat the L-1 training chunks for x (feature values) and y (class)
        x_chunk, y_chunk = [], []
        x_chunk.append(x_fold[:i] + x_fold[(i+1):])
        y_chunk.append(y_fold[:i] + y_fold[(i+1):])

        # training and validation sets concatenated into dataframes
        train_x, train_y = pd.concat(x_chunk[0]), pd.concat(y_chunk[0])
        validation_x, validation_y = x_fold[i], y_fold[i]

        # fitting the model
        model = KNN(K=K)
        model.fit(train_x, train_y)
        validation_prob, validation_knns = model.predict(validation_x)

        # choose class that has the max probability
        validation_pred = np.argmax(validation_prob, axis=-1)
        # y_test is a dataframe, convert it to a numpy array and calculate accuracy
        accuracy = model.evaluate_acc(validation_pred, validation_y['Class'].to_numpy())
        acc.append(accuracy)

    mean = round(np.mean(acc),5)
    var = round(np.var(acc),5)

    #print("\nKNN for K = ", K, " on given dataset")
    #print("Accuracies from cross validation: ", acc)
    #print("Mean of accuracies: ", mean)
    #print("Variance of accuracies: ", var)

    return mean, var

def DT_hyperparameter_tuning(dataset, train, features, L_fold):
    # split the class from the features
    x_train, y_train = train[features], train[['Class']]
    best_accuracy = 0.
    best_parameters = ""

    for combination in range(len(combinations)):
        print(combinations[combination])
        # implementing L-fold cross-validation on train data
        acc = []
        # splitting the training dataset into L chunks
        fold_size = int(train.shape[0] / L_fold)
        x_fold, y_fold = [], []

        for i in range(0, dataset.shape[0], fold_size):
            x_fold.append(x_train[i:(i + fold_size)])
            y_fold.append(y_train[i:(i + fold_size)])
        # the folds are created, iterate through and change validation set

        for i in range(L_fold):
            # need to concat the L-1 training chunks for x (feature values) and y (class)
            x_chunk, y_chunk = [], []
            x_chunk.append(x_fold[:i] + x_fold[(i + 1):])
            y_chunk.append(y_fold[:i] + y_fold[(i + 1):])

            # training and validation sets concatenated into dataframes
            train_x, train_y = pd.concat(x_chunk[0]), pd.concat(y_chunk[0])
            validation_x, validation_y = x_fold[i], y_fold[i]

            # fitting the model
            tree = decision_tree(max_depth=combinations[combination]['max_depth'],
                                cost_fn=combinations[combination]['cost_fn'],
                                min_leaf_instances=combinations[combination]['min_leaf_instances'])
            # tree = DecisionTree(max_depth=combinations[combination]['max_depth'])
            # tree = DecisionTree(cost_fn=combinations[combination]['cost_fn'])
            tree = decision_tree(min_leaf_instances=combinations[combination]['min_leaf_instances'])
            probs_test = tree.fit(x_train.to_numpy(), y_train.values.flatten()).predict(validation_x.to_numpy())
            y_pred = np.argmax(probs_test, 1)
            accuracy = tree.evaluate_acc(y_pred, validation_y.values.flatten())
            acc.append(accuracy)

        mean_accuracy = np.mean(acc)
        variance_accuracy = np.var(acc)
        print("Mean of accuracies: ", round(np.mean(acc), 2))
        # print("Variance of accuracies: ", round(np.var(acc), 2))

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_parameters = combinations[combination]

    print("Best accuracy: ", round(best_accuracy, 2))
    print("Best parameters: ", best_parameters)

    return best_parameters, best_accuracy

def decision_boundary():
    messidor_df = messidor_data()
    hep_df = hepatitis_data()

    hep_df.loc[:, 'Protime'] = hep_df['Protime'].apply(lambda x: x/10)

    # Hepatitis Data visualization
    f1, f2 = 'Albumin', 'Protime'  # change features to see the layout
    x1, y1 = hep_df[[f1, f2]], hep_df['Class']
    x1_train, y1_train = x1.iloc[:60], y1.iloc[:60]
    x1_test, y1_test = x1.iloc[60:], y1.iloc[60:]

    # Visualization of the data
    plt.scatter(x1_train.iloc[:, 0], x1_train.iloc[:, 1], c=y1_train, marker='o', label='train')
    plt.scatter(x1_test.iloc[:, 0], x1_test.iloc[:, 1], c=y1_test, marker='s', label='test')
    plt.legend()
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.show()

    # Messidor Data visualization
    f3, f4 = 'MA_Detection_0.8', 'MA_Detection_0.5'  # change features to see the layout
    x2, y2 = messidor_df[[f3, f4]], messidor_df['Class']
    x2_train, y2_train = x2.iloc[:1000], y2.iloc[:1000]
    x2_test, y2_test = x2.iloc[1000:], y2.iloc[1000:]

    plt.scatter(x2_train.iloc[:, 0], x2_train.iloc[:, 1], c=y2_train, marker='o', label='train')
    plt.scatter(x2_test.iloc[:, 0], x2_test.iloc[:, 1], c=y2_test, marker='s', label='test')
    plt.legend()
    plt.xlabel(f3)
    plt.ylabel(f4)
    plt.show()

    # Decision Boundary for two most highly correlated features
    # Hepatitis Data
    data = hep_df.sample(frac=1) # randomize
    features = ['Albumin', 'Protime']
    x, y = data[features], data[['Class']]
    (N, D), C = x.shape, y['Class'].max() + 1

    # split the dataset into train and test
    x_train, y_train = x[:60], y[:60]
    x_test, y_test = x[60:], y[60:]

    x0v = np.linspace(np.min(x.iloc[:, 0]), np.max(x.iloc[:, 0]), 200)
    x1v = np.linspace(np.min(x.iloc[:, 1]), np.max(x.iloc[:, 1]), 200)

    # to features values as a mesh
    x0, x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(), x1.ravel())).T
    x_all = pd.DataFrame(x_all)

    model = KNN(K=12)
    model.fit(x_train, y_train)

    y_train_prob = np.zeros((y_train.shape[0], 3))
    y_train_prob[np.arange(y_train.shape[0]), y_train] = 1

    # to get class probability of all the points in the 2D grid
    y_prob_all, _ = model.predict(x_all)

    y_pred_all = np.zeros_like(y_prob_all)
    y_pred_all[np.arange(x_all.shape[0]), np.argmax(y_prob_all, axis=-1)] = 1
    z = np.zeros((40000, 1))
    y_pred_all = np.append(y_pred_all, z, axis=1)

    plt.scatter(x_train[features[0]], x_train[features[1]], c=y_train_prob, marker='o', alpha=1)
    plt.scatter(x_all.iloc[:, 0], x_all.iloc[:, 1], c=y_pred_all, marker='.', alpha=0.01)
    plt.title('Decision Boundary for Albumin and Protime with K=12')
    plt.xlabel('Albumin')
    plt.ylabel('Protime')
    plt.show()

    # Hepatitis Data -Decision Tree
    data = hep_df.sample(frac=1)
    features = ['Albumin', 'Protime']
    x, y = data[features], data[['Class']]
    (N, D), C = x.shape, y['Class'].max() + 1

    inds = np.random.permutation(N)
    x_train, y_train = x.iloc[inds[:60]].to_numpy(), y.iloc[inds[:60]].values.flatten()
    x_test, y_test = x.iloc[inds[60:]].to_numpy(), y.iloc[inds[60:]].values.flatten()

    x0v = np.linspace(np.min(x.iloc[:, 0]), np.max(x.iloc[:, 0]), 200)
    x1v = np.linspace(np.min(x.iloc[:, 1]), np.max(x.iloc[:, 1]), 200)

    # to features values as a mesh
    x0, x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(), x1.ravel())).T
    # x_all = pd.DataFrame(x_all)

    model = decision_tree(max_depth=200)
    y_train_prob = np.zeros((y_train.shape[0], 3))
    y_train_prob[np.arange(y_train_prob.shape[0]), y_train] = 1
    y_prob_all = model.fit(x_train, y_train).predict(x_all)
    y_prob_all_draw = np.zeros((y_prob_all.shape[0], C + 1))

    for i in range(y_prob_all.shape[0]):
        y_prob_all_draw[i, 0:2] = y_prob_all[i, 0:2]

    # to plot
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_prob, marker='o', alpha=1)
    plt.scatter(x_all[:, 0], x_all[:, 1], c=y_prob_all_draw, marker='.', alpha=0.01)
    plt.title('Decision Boundary for Albumin and Protime with Decision Tree')
    plt.xlabel('Albumin')
    plt.ylabel('Protime')
    plt.show()

    # Decision Boundary for two most highly correlated features in messidor dataset
    data = messidor_df.sample(frac=1)
    features = ['MA_Detection_0.8', 'MA_Detection_0.5']
    x, y = data[features], data[['Class']]
    (N, D), C = x.shape, y['Class'].max() + 1
    train, test = data[:1000], data[1000:]

    x0v = np.linspace(np.min(train[features[0]]), np.max(train[features[0]]), 200)
    x1v = np.linspace(np.min(train[features[1]]), np.max(train[features[1]]), 200)

    # to features values as a mesh
    x0, x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(), x1.ravel())).T
    x_all = pd.DataFrame(x_all)

    model = KNN(K=40)
    model.fit(train[features], train[['Class']])
    # y_prob, knns = model.predict(test[features])
    # y_pred = np.argmax(y_prob, axis=-1)
    y_train_prob = np.zeros((train[['Class']].shape[0], C+1))
    y_train_prob[np.arange(train[['Class']].shape[0]), train[['Class']]] = 1

    # to get class probability of all the points in the 2D grid
    y_prob_all, _ = model.predict(x_all)
    y_prob_all_draw = np.zeros((y_prob_all.shape[0], C+1))

    for i in range(y_prob_all.shape[0]):
        y_prob_all_draw[i, 0:2] = y_prob_all[i, 0:2]

    plt.scatter(train[features[0]], train[features[1]], c=y_train_prob, marker='o', alpha=1)
    plt.scatter(x_all.iloc[:, 0], x_all.iloc[:, 1], c=y_prob_all_draw, marker='.', alpha=0.01)
    plt.title('Decision Boundary for MA Detection 0.8 and MA Detection 0.5 with K=40')
    plt.xlabel('MA Detection 0.8')
    plt.ylabel('MA Detection 0.5')
    plt.show()

def main():
    np.random.seed(123456)

    decision_boundary()

    data = hepatitis_data()
    data = messidor_data()
    # randomize the data
    data = data.sample(frac=1)
    data = data.sample(frac=1)

    features = ['Albumin', 'Protime']
    features = ['MA_Detection_0.5', 'MA_Detection_0.6', 'MA_Detection_0.7', 'MA_Detection_0.8', 'Exudates_0.99']
    # info on training and test set
    x, y = data[features], data[['Class']]
    (N, D), C = x.shape, y['Class'].max()+1
    print("instances (N) \t ", N, "\n features (D) \t ", D, " ", features, "\n classes (C) \t ", C)
    # split the data into train and test
    train, test = data[:60], data[60:]
    noise = [.01, .1, 1, 10, 100, 1000]
    K = 40
    L = 10
    means, variances, test_acc, K_list = [], [], [], []
    for i in range(len(noise)):#K in range(40, 41, 1):
        K_list.append(noise[i])
        data.loc[:, 'Exudates_0.99'] = data['Exudates_0.99'].apply(lambda x: x*noise[i])
        mean, var = KNN_cross_validation(data, train, features, K, L)
        means.append(mean)
        variances.append(var)

        # fit the model and calculate accuracy on unseen/test data
        model = KNN(K=K)
        model.fit(train[features], train[['Class']])
        prob, knns = model.predict(test[features])

        # choose class that has the max probability
        predictions = np.argmax(prob, axis=-1)
        accuracy = model.evaluate_acc(predictions, test['Class'].to_numpy())
        test_acc.append(accuracy)
        # put it back
        data.loc[:, 'Exudates_0.99'] = data['Exudates_0.99'].apply(lambda x: x/noise[i])

    plt.errorbar(K_list, means, variances, label='validation')
    plt.plot(K_list, test_acc, label='test')
    plt.legend()
    plt.title('Accuracies with 10-fold CV, euclidean distance and K=40')
    plt.xlabel('Scale of noisy feature')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    main()