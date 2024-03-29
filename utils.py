import os
import numpy as np
import pandas as pd
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

def read_text(path):
    """Read article.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    text : str
        The text of the article.
    """
    #print(path)
    text = np.nan
    try:
        with open(path, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]
        lines = list(filter(lambda x: x != "", lines))

        text = " ".join(lines)
    except Exception as e:
        print(f"skipping {path} due to: {e}")
    finally:
        return text


class DataReader():
    def __init__(self, data_path, spark):
        """Init.

        Parameters
        ----------
        data_path : str
            Path to the data folder.
        spark : pyspark.sql.session.SparkSession
            Spark session.
        """
        self.data_path = data_path
        self.spark = spark

    def __call__(self, topic_names):
        """Read data.

        Parameters
        ----------
        topic_names : list
            List of topics among articles.

        Returns
        -------
        df_data_all : spark.DataFrame
            DataFrame of data with columns: path (path to file), topic (topic of article e.g.: business),
            text (the text of the article), id (unique id of the article).
        """
        data = {}
        paths = []
        topics = []
        texts = []

        for topic_name in topic_names:
            dir_path = os.path.join(self.data_path, topic_name)

            for dirpath, dirnames, filenames in os.walk(dir_path):
                paths_ = [os.path.join(dirpath, filename) for filename in filenames]
                topics_ = [topic_name for i in range(len(paths_))]

                texts_ = [read_text(path_) for path_ in paths_]

                paths += paths_
                topics += topics_
                texts += texts_

        data["path"] = paths
        data["topic"] = topics
        data["text"] = texts
        data["id"] = [i for i in range(len(texts))]

        pd_df = pd.DataFrame(data=data)
        pd_df.dropna(inplace=True)

        df_data_all = self.spark.createDataFrame(pd_df, list(pd_df.columns.values))

        return df_data_all


def make_heatmap(rdd_collected, title, epoch_time):
    my_cmap = copy.copy(plt.cm.get_cmap('plasma'))
    my_cmap.set_bad((0, 0, 0))

    # rdd_collected = rdd.collect()
    n = len(rdd_collected)
    heatmap = np.zeros((n, n))

    for idx_l, l in enumerate(rdd_collected):
        for idx_t, t in enumerate(l):
            idx, s = t
            assert idx_t == idx
            heatmap[idx_l, idx_t] = s

    plt.figure(figsize=(10, 10))

    sns.heatmap(heatmap, square=True, norm=LogNorm(), cmap=my_cmap)

    plt.title(title)
    fname = title + "_" + str(epoch_time) + ".png"
    plt.savefig(f"assets/{fname}")
    plt.plot()
