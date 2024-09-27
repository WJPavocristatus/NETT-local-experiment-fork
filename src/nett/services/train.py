import os
import glob
import numpy as np
import pandas as pd

def compute_train_performance(path) -> tuple[list, np.ndarray | list]:

    x,y = [], []
    try:
        training_files = glob.glob(os.path.join(path, "*.csv"))

        if len(training_files) == 0:
            raise Exception(f"Training file: {training_files} was not found in the {path}")


        for file_name in training_files:

            log_df = pd.read_csv(file_name, skipinitialspace=True)

            _, _, values = average_in_episode_three_region(log_df,"agent.x") # percents,df,
            y = moving_average(values, window=100)
            x = list(range(len(y)))

            break


        return x, y
    except Exception as ex:
        print(str(ex))

    return x,y

def average_in_episode_three_region(log: pd.DataFrame, column: str = 'agent.x', transient: int = 90) -> tuple[dict, pd.DataFrame, list]:
    try:
        log.loc[log.Episode % 2 == 1, column] *= -1
        #Translate coordinates
        log[column] += 10
        #Bin into 3 sections
        log[column] = pd.cut(log[column], [-0.1,20/3,40/3,20.1],labels=["Distractor","Null","Imprint"])
        episodes = log.Episode.unique()
        percents = {}
        for ep in episodes:
            #Get success percentage
            l = log[log["Episode"]==ep]
            l = l[l["Step"]>transient]
            total = l[l[column]=="Distractor"].count() + l[l[column]=="Imprint"].count()
            success = l[l[column]=="Imprint"].count()/total
            percents[ep] = success[column]

            if np.isnan(percents[ep]):
                percents[ep] = 0.5

        rv = list(percents.values())

        return (percents,log,rv)
    except Exception as ex:
        print(str(ex))
        return (None, None, None)

def moving_average(values: list, window: int) -> np.ndarray:
    weights: np.ndarray = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')
