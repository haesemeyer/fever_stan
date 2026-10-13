# Utility functions
import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from os import path
import os
import argparse


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'folder':
            if not path.exists(values):
                raise argparse.ArgumentError(self, f"Specified directory {values} does not exist")
            if not path.isdir(values):
                raise argparse.ArgumentError(self, "The destination is a file but should be a directory")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def find_all_pickle_paths(root_f: str) -> List[str]:
    """
    Recursively searches for pandas pickle files identified by their .pk extension
    :param root_f: The folder from which to start the current search
    :return: List of all .pk files at and below root_f with their full path
    """
    try:
        objects = os.listdir(root_f)
    except PermissionError:
        return []
    exp_local = [path.join(root_f, o) for o in objects if ".pk" in o and "fish" not in o]
    dir_local = [path.join(root_f, o) for o in objects if path.isdir(path.join(root_f, o))]
    exp_deep = []
    for dl in dir_local:
        e_deep = find_all_pickle_paths(dl)
        exp_deep += e_deep
    return exp_local + exp_deep


def load_all_trajectories(folder: str, len_thresh=3) -> List[pd.DataFrame]:
    """
    Load all trajectory information from pandas pickle files in the given folder
    :param folder: The folder to search for files
    :param len_thresh: Remove all trajectories with a length less than this value
    :return: Aggregated list of all trajectories
    """
    pickle_files = find_all_pickle_paths(folder)
    pickle_files = sorted(pickle_files)
    trajectories = []
    for pf in pickle_files:
        temp_npy_file = path.splitext(pf)[0] + "_activity.npy"
        if path.exists(temp_npy_file):
            bout_activity = np.load(temp_npy_file)
        else:
            bout_activity = None
        df = pd.read_pickle(pf)
        if bout_activity is not None:
            assert bout_activity.shape[0] == df.shape[0]
            # insert cluster activity for each bout
            for clix in range(bout_activity.shape[1]):
                df.insert(0, f"Cluster activity_{clix}", bout_activity[:, clix], False)
        possibles = collect_trajectories(df)
        trajectories += [p for p in possibles if p.shape[0] >= len_thresh]
    return trajectories


def collect_trajectories(df: pd.DataFrame, frame_rate=100, n_seconds=2) -> List[pd.DataFrame]:
    """
    From one bout structure generates a list of bout structures with each structure representing
    one continuous trajectory
    :param df: A bout dataframe
    :param frame_rate: To convert bout distances into time for trajectory breaking
    :param n_seconds: Maximal delta-t between consecutive bout starts within a trajectory
    :return: List of trajectory bout dataframes
    """
    traj_breaks = np.r_[1, np.diff(df['Original index'])] > 1
    # add additional trajectory breaks for bouts that are more than n_seconds apart
    delta_time = np.r_[0, np.diff(df['Start'])] / frame_rate
    traj_breaks = np.logical_or(traj_breaks, delta_time > n_seconds)
    traj_starts = np.r_[0, np.where(traj_breaks)[0], traj_breaks.size]
    traj_list = []
    for i, ts in enumerate(traj_starts[:-1]):
        df_traj = df.iloc[ts:traj_starts[i+1]]
        # add in previous angle, shortening trajectory by one - NOTE: We have to do this hear rather than on the
        # original bout dataframe since the original bout dataframe has already been edge filtered!!!
        prev_angles = np.array(df_traj["Angle change"][:-1])
        prev_displace = np.array(df_traj["Displacement"][:-1])
        df_traj = df_traj[1:]
        assert df_traj.shape[0] == prev_angles.size  # don't trust pandas
        df_traj.insert(0, "Prev angle", prev_angles, False)
        df_traj.insert(0, "Prev displacement", prev_displace, False)
        traj_list.append(df_traj)
    return traj_list


def split_train_test(trajectories: List[pd.DataFrame], test_every=5) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Split trajectories into train and test sets
    :param trajectories: The experimental trajectories as pandas dataframes
    :param test_every: Every test_every entry in trajectories will be added to the test-set, remaining is train
    :return:
        [0]: List with training trajectories
        [1]: List with testing trajectories
    """
    train_traj, test_traj = [], []
    for i, traj in enumerate(trajectories):
        if i % test_every == 0:
            test_traj.append(traj)
        else:
            train_traj.append(traj)
    return train_traj, test_traj
