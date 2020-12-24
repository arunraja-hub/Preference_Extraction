import os

import numpy as np

from data_getter import get_data_from_file, get_data_from_folder, write_file

DOOM_HUMAN_MAX_ANGLE = 90
DOOM_HUMAN_MIN_ANGLE = 270

def transform_to_x_y(raw_data, env):
    """
        Takes in raw trajectories data and returns processed xs and ys
    """
    xs, ys = [], []

    for data in raw_data:
        for i in range(data.observation.shape[0]):
            x = np.copy(data.observation[i])

            if env.lower() == 'doom':
                # Doom label object is a dictionary with object_angle and distance_from_wall
                label_object = data.policy_info['satisfaction'][i]
                if len(label_object) == 0: # When label is empty, i.e. human is dead, skip frame
                    continue
                else:
                    y = label_object['object_angle'] < DOOM_HUMAN_MAX_ANGLE \
                        or label_object['object_angle'] > DOOM_HUMAN_MIN_ANGLE
            else:
                y = data.policy_info['satisfaction'].as_list()[i] > -6

            xs.append(x)
            ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys).astype(int)
    print("xs", xs.shape, "ys", ys.shape)
    print("ys 1", np.sum(ys))
    return xs, ys
    
def rebalance_data_to_minority_class(xs, ys):
    """
       Specific for binary labels. Assumes that 1s are the minority class
       Gets a random perumtation of 0s that include a number of indexes equals to the 1s
       and concatenates them with the 1s
    """
    original_ys = ys
    
    indexes1 = [i for i in range(len(xs)) if ys[i] == 1]
    indexes0 = [i for i in range(len(xs)) if ys[i] == 0]
    x0, x1, y0, y1 = xs[indexes0], xs[indexes1], ys[indexes0], ys[indexes1]
    
    sample_ind = np.random.permutation(len(y0))[:len(y1)]    
    x0, y0 = x0[sample_ind], y0[sample_ind] # sub-samples the 0s

    xs = np.concatenate((x0, x1))
    ys = np.concatenate((y0, y1))
    
    print('Data rebalanced from', original_ys.shape, 'to', ys.shape)
    
    return xs, ys

def data_pipeline(data_path, from_file=True, env='doom', rebalance=True):
    if from_file:
        data = get_data_from_file(data_path)
    else: # for data as list of trajectory files
        data = get_data_from_folder(data_path)

    xs, ys = transform_to_x_y(data, env=env)
    if rebalance:
        xs, ys = rebalance_data_to_minority_class(xs, ys)
    
    return xs, ys

def reshuffle_data(xs, ys):
    randomize = np.arange(len(xs))
    np.random.shuffle(randomize)
    xs = xs[randomize]
    ys = ys[randomize]
    return xs, ys

if __name__ == "__main__":
    
    # pre-training data creation
    data_path = "gs://pref_extract_train_output/ppo_search_log_fix_1455626/10/exp_data_20000.pkl"
    from_file = True
    env = "Doom"
    rebalance = True
    data_version = '1'
    gcs_bucket = 'gs://pref-extr-data/{}/'.format(env.lower())
    
    xs, ys = data_pipeline(data_path=data_path, from_file=from_file, env=env, rebalance=rebalance)
    xs, ys = reshuffle_data(xs, ys)
    
    # create train_train, train_validate, test_train, test_validate datasets
    trn_size = 50
    val_size = 500
    no_of_train_runs = 10
    no_of_test_runs = 2
    
    xs_train = xs[:(trn_size+val_size)*no_of_train_runs]
    ys_train = ys[:(trn_size+val_size)*no_of_train_runs]
    xs_test = xs[(trn_size+val_size)*no_of_train_runs:(trn_size+val_size)*(no_of_train_runs+no_of_test_runs)]
    ys_test = ys[(trn_size+val_size)*no_of_train_runs:(trn_size+val_size)*(no_of_train_runs+no_of_test_runs)]
    
    print('Train shapes', xs_train.shape, ys_train.shape)
    print('Test shapes', xs_test.shape, ys_test.shape)
    
    # save data to GCS
    train_path = os.path.join(gcs_bucket, 'data', 'train_val_data_{}.pkl'.format(data_version))
    test_path = os.path.join(gcs_bucket, 'data', 'test_val_data_{}.pkl'.format(data_version))
    write_file(train_path, (xs_train, ys_train))
    write_file(test_path, (xs_test, ys_test))