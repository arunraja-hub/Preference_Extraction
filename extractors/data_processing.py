import numpy as np

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
                    y = label_object['object_angle'] < 90 or label_object['object_angle'] > 270
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
    
    # Rebalancing data to minority class
    points = xs
    labels = ys

    # indexes of 1s and 0s
    indexes1 = [i for i in range(len(points)) if labels[i] == 1]
    indexes0 = [i for i in range(len(points)) if labels[i] == 0]

    # separate 0s and 1s
    x0, x1, y0, y1 = points[indexes0], points[indexes1], labels[indexes0], labels[indexes1]

    minority_points, minority_labels = x1, y1  # points and labels for the minority class
    majority_points, majority_labels = x0, y0  # points and labels for the majority class

    # get a random permutation of indexes of the majority that includes a number of indexes equal to the minority
    sample_ind = np.random.permutation(len(majority_labels))[:len(minority_labels)]

    # subsample the majority
    majority_points, majority_labels = majority_points[sample_ind], majority_labels[sample_ind]

    # concat the minority and the sub-sampled majority
    xs = np.concatenate((majority_points, minority_points))
    ys = np.concatenate((majority_labels, minority_labels))
    
    print('Data rebalanced from', labels.shape, 'to', ys.shape)
    
    return xs, ys