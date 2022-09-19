import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import OneHotEncoder


def minmax_normalizer(a, min_a=None, max_a=None, correction=False):
    if min_a is None:
        min_a = np.min(a, 0)
    if max_a is None:
        max_a = np.max(a, 0)
    if correction:
        scale = (a - min_a) / (max_a - min_a + 0.0001)
    else:
        scale = (a - min_a) / (max_a - min_a)
    return scale, min_a, max_a


def load_data(dataset_type, src_path, dest_path, one_hot=False, remove_one_val=False):
    os.makedirs(dest_path, exist_ok=True)

    if dataset_type == 'SWaT':
        train_file = os.path.join(src_path, 'SWaT_Dataset_Normal_v0.csv')
        test_file = os.path.join(src_path, 'SWaT_Dataset_Attack_v0.csv')
        df_train = pd.read_csv(train_file, index_col=0)
        print(df_train)
        df_test = pd.read_csv(test_file, index_col=0)
        df_train.columns = [i.strip() for i in df_train.columns.values.tolist()]
        print(df_train.columns)
        df_test.columns = [i.strip() for i in df_test.columns.values.tolist()]
        print(df_test.columns)
        label_name = 'Normal/Attack'

        df_train.drop(['Timestamp', label_name], axis=1, inplace=True)
        labels = [float(label != 'Normal') for label in df_test[label_name].values]
        df_test.drop(['Timestamp', label_name], axis=1, inplace=True)

        anomaly_ratio = np.unique(labels, return_counts=True)[1] / len(labels)

        print('Train shape:', df_train.shape)
        print('Test shape:', df_test.shape)
        print('Anomaly ratio:', np.round(anomaly_ratio[1], 4))

        for col in df_train:
            if isinstance(df_train[col].iloc[0], str) or col == 'AIT401':
                if col == 'Normal/Attack':
                    pass
                else:
                    df_train[col] = [float(str(v).replace(',', '.')) for v in df_train[col].values]

        for col in df_test:
            if isinstance(df_test[col].iloc[0], str):
                if col == 'Normal/Attack':
                    pass
                else:
                    df_test[col] = [float(str(v).replace(',', '.')) for v in df_test[col].values]

        if remove_one_val:
            train_idx = np.where(df_train.apply(lambda x: len(np.unique(x))) != 1)[0]
            df_train = df_train.iloc[:, train_idx]
            df_test = df_test.iloc[:, train_idx]

        if one_hot:
            swat_discrete = ["MV101", "MV201", "MV301", "MV302", "MV303", "MV304",
                             "P101", "P102", "P201", "P202", "P203", "P204", "P205", "P206",
                             "P301", "P302", "P401", "P402", "P403", "P404",
                             "P501", "P502", "P601", "P602", "P603", "UV401"]
            swat_discrete = [i in swat_discrete for i in df_train.columns.values]
            swat_continuous = np.logical_not(swat_discrete)
            df_train_d = df_train.iloc[:, swat_discrete]
            df_train_c = df_train.iloc[:, swat_continuous]
            df_test_d = df_test.iloc[:, swat_discrete]
            df_test_c = df_test.iloc[:, swat_continuous]

            train_c, train_min, train_max = minmax_normalizer(df_train_c.values)
            test_c, _, _ = minmax_normalizer(df_test_c.values, train_min, train_max)

            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(df_train_d.values)
            train_d = enc.transform(df_train_d.values)
            train_d = train_d.toarray()

            test_d = enc.transform(df_test_d.values)
            test_d = test_d.toarray()

            train = np.hstack([train_c, train_d])
            test = np.hstack([test_c, test_d])

        else:

            if remove_one_val:
                train, train_min, train_max = minmax_normalizer(df_train.values)
                test, _, _ = minmax_normalizer(df_test.values, train_min, train_max)
            else:
                train, train_min, train_max = minmax_normalizer(df_train.values, correction=True)
                test, _, _ = minmax_normalizer(df_test.values, train_min, train_max, correction=True)

        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(dest_path, f'{file}.npy'), eval(file))
        with open(os.path.join(dest_path, 'list.txt'), "w") as file:
            file.writelines(df_train.columns.values + '\n')

    elif dataset_type == 'WADI':
        wadi_drop = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']

        train_file = os.path.join(src_path, 'WADI_14days_new.csv')
        test_file = os.path.join(src_path, 'WADI_attackdataLABLE.csv')
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file, header=1)

        df_train.columns = [i.strip() for i in df_train.columns.values.tolist()]
        df_test.columns = [i.strip() for i in df_test.columns.values.tolist()]

        df_train.drop(wadi_drop, axis=1, inplace=True)
        df_test.drop(wadi_drop, axis=1, inplace=True)
        df_train.dropna(axis=0, inplace=True)
        df_test.dropna(axis=0, inplace=True)

        label_name = 'Normal/Attack'
        df_test.rename(columns={df_test.columns[-1]: label_name}, inplace=True)
        df_test[label_name].replace({1: 0, -1: 1}, inplace=True)

        labels = df_test[label_name]

        df_train.drop(['Row', 'Date', 'Time'], axis=1, inplace=True)
        df_test.drop(['Row', 'Date', 'Time', label_name], axis=1, inplace=True)
        anomaly_ratio = np.unique(labels, return_counts=True)[1] / len(labels)

        print('Train shape:', df_train.shape)
        print('Test shape:', df_test.shape)
        print('Anomaly ratio:', np.round(anomaly_ratio[1], 4))

        if remove_one_val:
            train_idx = np.where(df_train.apply(lambda x: len(np.unique(x))) != 1)[0]
            df_train = df_train.iloc[:, train_idx]
            df_test = df_test.iloc[:, train_idx]

        if one_hot:

            wadi_discrete = ["1_MV_001_STATUS", "1_MV_002_STATUS", "1_MV_003_STATUS",
                             "1_MV_004_STATUS", "2_MV_001_STATUS", "2_MV_002_STATUS",
                             "2_MV_003_STATUS", "2_MV_004_STATUS", "2_MV_005_STATUS",
                             "2_MV_006_STATUS", "2_MV_009_STATUS", "2_MV_101_STATUS",
                             "2_MV_201_STATUS", "2_MV_301_STATUS", "2_MV_401_STATUS",
                             "2_MV_501_STATUS", "2_MV_601_STATUS", "3_MV_001_STATUS",
                             "3_MV_002_STATUS", "3_MV_003_STATUS", "1_P_001_STATUS",
                             "1_P_002_STATUS", "1_P_003_STATUS", "1_P_004_STATUS",
                             "1_P_005_STATUS", "1_P_006_STATUS",
                             "2_P_003_STATUS", "2_P_004_STATUS",
                             "3_P_001_STATUS", "3_P_002_STATUS", "3_P_003_STATUS", "3_P_004_STATUS",
                             "1_LS_001_AL", "1_LS_002_AL", "2_LS_101_AH",
                             "2_LS_101_AL", "2_LS_201_AH", "2_LS_201_AL", "2_LS_301_AH",
                             "2_LS_301_AL", "2_LS_401_AH", "2_LS_401_AL", "2_LS_501_AH",
                             "2_LS_501_AL", "2_LS_601_AH", "2_LS_601_AL", "3_LS_001_AL",
                             "2_SV_101_STATUS", "2_SV_201_STATUS", "2_SV_301_STATUS",
                             "2_SV_401_STATUS", "2_SV_501_STATUS", "2_SV_601_STATUS"]

            wadi_discrete = [i in wadi_discrete for i in df_train.columns.values]
            wadi_continuous = np.logical_not(wadi_discrete)
            df_train_d = df_train.iloc[:, wadi_discrete]
            df_train_c = df_train.iloc[:, wadi_continuous]
            df_test_d = df_test.iloc[:, wadi_discrete]
            df_test_c = df_test.iloc[:, wadi_continuous]

            train_c, train_min, train_max = minmax_normalizer(df_train_c.values)
            test_c, _, _ = minmax_normalizer(df_test_c.values, train_min, train_max)

            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(df_train_d.values)
            train_d = enc.transform(df_train_d.values)
            train_d = train_d.toarray()

            test_d = enc.transform(df_test_d.values)
            test_d = test_d.toarray()

            train = np.hstack([train_c, train_d])
            test = np.hstack([test_c, test_d])

        else:

            if remove_one_val:
                train, train_min, train_max = minmax_normalizer(df_train.values)
                test, _, _ = minmax_normalizer(df_test.values, train_min, train_max)
            else:
                train, train_min, train_max = minmax_normalizer(df_train.values, correction=True)
                test, _, _ = minmax_normalizer(df_test.values, train_min, train_max, correction=True)

        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(dest_path, f'{file}.npy'), eval(file))

        with open(os.path.join(dest_path, 'list.txt'), "w") as file:
            file.writelines(df_train.columns.values + '\n')

    elif dataset_type == 'SMD':
        file_list = os.listdir(os.path.join(src_path, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                train = np.genfromtxt(os.path.join(src_path, 'train', filename), dtype=np.float64, delimiter=',')
                test = np.genfromtxt(os.path.join(src_path, 'test', filename), dtype=np.float64, delimiter=',')
                labels = np.genfromtxt(os.path.join(src_path, 'test_label', filename), dtype=np.float64,
                                       delimiter=',')

                train, train_min, train_max = minmax_normalizer(train, correction=True)
                test, _, _ = minmax_normalizer(test, min_a=train_min, max_a=train_max, correction=True)

                fn = filename.strip('.txt')
                p = train.shape[1]
                column_name = np.array(["V" + str(i + 1) for i in range(p)], dtype=object)

                with open(os.path.join(dest_path, fn + "_list.txt"), "w") as file:
                    file.writelines(column_name + '\n')

                df_train = pd.DataFrame(train, columns=column_name)
                df_test = pd.DataFrame(test, columns=column_name)
                anomaly_ratio = np.unique(labels, return_counts=True)[1] / len(labels)

                print(fn, '| Train shape:', df_train.shape)
                print(fn, '| Test shape:', df_test.shape)
                print(fn, '| Anomaly ratio:', np.round(anomaly_ratio[1], 4))

                if remove_one_val:
                    train_idx = np.where(df_train.apply(lambda x: len(np.unique(x))) != 1)[0]
                    df_train = df_train.iloc[:, train_idx]
                    df_test = df_test.iloc[:, train_idx]

                np.save(f'{dest_path}/{fn}_train.npy', df_train.values)
                np.save(f'{dest_path}/{fn}_test.npy', df_test.values)
                np.save(f'{dest_path}/{fn}_labels.npy', labels)


    elif dataset_type == 'SMAP' or dataset_type == 'MSL':
        file = os.path.join(src_path, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset_type]
        filenames = values['chan_id'].values.tolist()
        filenames = [channel for channel in filenames if channel != 'P-2']

        for fn in filenames:
            train = np.load(f'{src_path}/train/{fn}.npy')
            test = np.load(f'{src_path}/test/{fn}.npy')

            train, train_min, train_max = minmax_normalizer(train, correction=True)
            test, _, _ = minmax_normalizer(test, train_min, train_max, correction=True)

            labels2 = np.zeros(test.shape[0])
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels2[indices[i]:indices[i + 1] + 1] = 1

            p = train.shape[1]
            column_name = np.array(["V" + str(i + 1) for i in range(p)], dtype=object)

            df_train = pd.DataFrame(train, columns=column_name)
            df_test = pd.DataFrame(test, columns=column_name)
            anomaly_ratio = np.unique(labels2, return_counts=True)[1] / len(labels2)

            print(fn, '| Train shape:', df_train.shape)
            print(fn, '| Test shape:', df_test.shape)
            print(fn, '| Anomaly ratio:', np.round(anomaly_ratio[1], 4))

            if remove_one_val:
                train_idx = np.where(df_train.apply(lambda x: len(np.unique(x))) != 1)[0]
                df_train = df_train.iloc[:, train_idx]
                df_test = df_test.iloc[:, train_idx]

            np.save(f'{dest_path}/{fn}_train.npy', df_train.values)
            np.save(f'{dest_path}/{fn}_test.npy', df_test.values)
            np.save(f'{dest_path}/{fn}_labels.npy', labels2)

            with open(os.path.join(dest_path, fn + "_list.txt"), "w") as file:
                file.writelines(column_name + '\n')


if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')


    parser = argparse.ArgumentParser('preprocess')
    # parser.add_argument('--dataset', type = str, required=True)
    # parser.add_argument('--path', type = str, required=True)
    parser.add_argument('--one_hot', type=str2bool, default='False')
    parser.add_argument('--remove_ov', type=str2bool, default='False')

    arg_of_parser = parser.parse_args()

    # dataset = arg_of_parser.dataset
    # data_path = arg_of_parser.path
    one_hot = False
    remove_ov = False
    dataset_type = 'SWaT'
    src_path = './SWaT'
    dest_path = './SWaT'
    load_data(dataset_type, src_path, dest_path, one_hot, remove_ov)
