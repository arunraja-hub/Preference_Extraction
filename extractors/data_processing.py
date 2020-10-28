import numpy as np

DOOM_HUMAN_MAX_ANGLE = 90
DOOM_HUMAN_MIN_ANGLE = 270

def transform_to_x_y(raw_data, env, shuffle_data = True):
    """
        Takes in raw trajectories data and returns processed xs and ys
    """
    xs, ys = [], []

    for data in raw_data:
        for i in range(data.observation.shape[0]):
            x = np.copy(data.observation[i])

            if env == 'doom':
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
    
    if shuffle_data:
        randomize = np.arange(len(xs))
        np.random.shuffle(randomize)
        xs = xs[randomize]
        ys = ys[randomize]
        
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