import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def apply_moving_average(df: pd.DataFrame, column_names: list, window_size: int = 5, smooth_anomalies=False):
    df = df.copy()

    if smooth_anomalies:
        for column_name in column_names:
            df[column_name] = df[column_name].rolling(
                window=window_size, center=True, min_periods=1).mean()
    else:
        for column_name in column_names:
            # Only smooth non-anomalous data
            smooth_values = df.loc[df.anomaly == 0, column_name].rolling(
                window=window_size, center=True, min_periods=1).mean()
            
            # Assign smoothed values only to normal data points
            df.loc[df.anomaly == 0, column_name] = smooth_values
    
    return df


import plotly.graph_objects as go
def plot_smoothed_signal(df, column_names):
    fig = go.Figure()
    
    for column in column_names:
        # Original signal (optional)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=f"Smoothed: {column}"
        ))

    fig.update_layout(
        title="Smoothed Signal Plot",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(x=0, y=1)
    )
    
    fig.show()


def get_scaled_data_values(csv_path: str):
    # ddf = dd.read_csv(csv_path)

    columns = [col for col in ddf.columns if col.startswith('channel')]
    for column_name in columns:
        # ddf = scale_values(ddf, column_name)
        ddf = ddf.map_partitions(apply_moving_average, column_name, window_size=100)

    df = ddf.compute()

    return df, columns

        
def extract_signal_and_anomaly_array(df, columns: list):
    signals = []
    for col in columns:
        signal = df[col].values
        signals.append(signal)

    # Convert list of arrays to 2D array: shape (num_samples, num_channels)
    signals = np.stack(signals, axis=1)

    signal_df = pd.DataFrame(signals, columns=[f"{col}_scaled" for col in columns])

    # Assume a single 'anomaly' column applies to all
    labels = df["anomaly"].values

    return signal_df, labels


def crop_datetime(df, start_datetime="", end_datetime="", print_data_info=False):

    if print_data_info:
        print(df.head())
        print(df.info())
        print(df.describe())

    start_dt = pd.to_datetime(start_datetime, utc=True)
    end_dt  = pd.to_datetime(end_datetime, utc=True)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        data_filtered = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        data_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]
    
    return data_filtered

class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class ESASegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size

        filename = os.path.basename(data_path)
        channel = os.path.splitext(filename)[0]

        #df, channel_columns = get_scaled_data_values(data_path)

        df = pd.read_csv(data_path)

        # Train
        start_datetime="2001-01-01T00:00:00.000Z"
        end_datetime = "2001-10-01T00:00:00.000Z"
        filtered_df = crop_datetime(df, start_datetime, end_datetime)

        self.train = filtered_df[channel]

        # valiation
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
         
        # test
        start_datetime="2004-01-01T00:00:00.000Z"
        end_datetime = "2005-01-01T00:00:00.000Z"
        filtered_df = crop_datetime(df, start_datetime, end_datetime)

        self.test, self.test_labels = extract_signal_and_anomaly_array(filtered_df, channel)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class ESAPhaseSegLoader(object):
    def __init__(self, data_path, win_size, step, mode:str="train", phase=1):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.phase = phase

        #df, channel_columns = get_scaled_data_values(data_path)

        df = pd.read_csv(data_path)

        channel_columns = [col for col in df.columns if col.startswith('channel')]

        start_datetime = pd.Timestamp("2000-01-01T00:00:00.000Z")
        end_datetime = pd.Timestamp("2000-05-01T00:00:00.000Z")
        months_to_add = 4

        # PHASED TRAINING PERIODS
        if self.mode == "train":
            start_datetime = start_datetime + pd.DateOffset(months=months_to_add*(phase-1))
            end_datetime = end_datetime + pd.DateOffset(months=months_to_add*(phase-1))
            print(start_datetime)
            print(end_datetime)

            filtered_df = crop_datetime(df, start_datetime.isoformat(), end_datetime.isoformat())
            #smoothed_df = apply_moving_average(filtered_df, channel_columns, self.win_size, smooth_anomalies=True)
            #plot_smoothed_signal(smoothed_df, channel_columns)
            self.train = filtered_df[channel_columns]

        start_datetime = "2001-01-01T00:00:00.000Z"
        end_datetime = "2002-01-01T00:00:00.000Z"
        filtered_df = crop_datetime(df, start_datetime, end_datetime)
        smoothed_df = apply_moving_average(filtered_df, channel_columns, self.win_size, smooth_anomalies=True)
        self.test, self.test_labels = extract_signal_and_anomaly_array(smoothed_df, channel_columns)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                            index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', phase=1):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ESA'):
        dataset = ESASegLoader(data_path, win_size, step, mode)
    elif (dataset == 'ESAPhase'):
        dataset = ESAPhaseSegLoader(data_path, win_size, step, mode, phase)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
