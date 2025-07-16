import pandas as pd
import deeplenstronomy.deeplenstronomy as dl

all_params = ["PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g", "PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g", "PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g", "PLANE_1-OBJECT_1-MASS_PROFILE_1-center_x-g", "PLANE_1-OBJECT_1-MASS_PROFILE_1-center_y-g"] # have to update for 12param. but right now contains 1p,5p
def load(params):
    """take in param count (1,5 currently) and return (1) ground truth for target params (2) images"""
    dataset = dl.make_dataset(f"./data/{params}param_test_set.yaml")
    params_df = dataset.CONFIGURATION_1_metadata[all_params[0:params]] # extract relevant parameters
    return params_df, dataset.CONFIGURATION_1_images