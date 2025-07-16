import deeplenstronomy.deeplenstronomy as dl
def return_dataset(params):
    return dl.make_dataset(f"./data/{params}param_test_set.yaml")
