import numpy as np


class Shingling():
    """A class Shingling that constructs k–shingles of a given length k (e.g., 10) from a given document,
    computes a hash value for each unique shingle, and represents the document in the form of an
    ordered set of its hashed k-shingles.
    """
    def __init__(self, k):
        """Init.

        Parameters
        ----------
        k : int
            k-shingles.
        """
        self.k = k

    def shingles(self, text):
        """k-shingle text.

        Parameters
        ----------
        text : str
            Text as str.

        Returns
        -------
        shingles : list
            List (list of a set) of shingles in the text.
        """
        shingles = list(set([text[i:i + self.k] for i in range(len(text) - self.k + 1)]))
        return shingles

    def hash_shingle(self, shingle):
        """Hash a shingle.

        Parameters
        ----------
        shingle : str
            A shingle, a text of lenght k.

        Returns
        -------
        int
            Hashed shingle
        """
        return hash(shingle)

    def hash_singles(self, shingles):
        """Hash a shingles.

        Parameters
        ----------
        shingles : list
            Shingles are a list of strs, where each str is a text of lenght k.

        Returns
        -------
        list
            Hashed shingles.
        """
        return [self.hash_shingle(shingle) for shingle in shingles]

    def __call__(self, texts_rdd):
        """k-shingle a list of texts.

        Parameters
        ----------
        texts_rdd : pyspark.rdd.PipelinedRDD
            List of article texts.

        Returns
        -------
        k_shingles_rdd : pyspark.rdd.PipelinedRDD
            A list of (hashed) k-shingles for each text.
        k_shingles_all_rdd : pyspark.rdd.PipelinedRDD
            A list of all of the (hashed) k-shingles across all of the texts.
        """
        # convert texts to k-grams per document
        k_grams_rdd = texts_rdd.map(lambda x: self.shingles(x))
        # hash k-grams per document to get k-shingles per document
        k_shingles_rdd = k_grams_rdd.map(lambda x: self.hash_singles(x))

        # k-grams across all documents, flat
        k_grams_all_rdd = k_grams_rdd.flatMap(lambda x: x)
        # k-grams and their counts across all docs
        k_grams_counts_rdd = k_grams_all_rdd.map(lambda w: (w ,1)).reduceByKey(lambda a, b: a+ b)

        # hashed k-grams across all documents, flat
        hashed_k_grams_all_rdd = k_shingles_rdd.flatMap(lambda x: x)
        # getting the set of all hashed k grams across docs, keys and counts
        hashed_k_grams_all_reduced_counts_rdd = \
            hashed_k_grams_all_rdd.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)
        # keys only, below is a list of all hashed k-grams across the docs
        k_shingles_all_rdd = hashed_k_grams_all_reduced_counts_rdd.map(lambda x: x[0])

        return k_shingles_rdd, k_shingles_all_rdd


class CompareSets():
    """A class CompareSets that computes the Jaccard similarity of two sets of integers – two sets
    of hashed shingles."""
    def __init__(self, ):
        pass

    def vectorize(self, shingles, shingles_all):
        """Make Boolean vector of hashed k-shingles per text such that an entry in the vector is True if the given
        hashed shingle is in the text, False otherwise. All vectors are len(shingles_all) long. shingles_all is ordered.

        Parameters
        ----------
        shingles : list
            List of hashed k-shingles per text.
        shingles_all : list
            List of hashed k-shingles across texts.

        Returns
        -------
        list
            List of Booleans, True if a shingle is present in the text, False otherwise. Order matters.
        """
        return [shingle in shingles for shingle in shingles_all]

    def compare_sets(self, vec, vec_all):
        """Compare vectorized sets of k-shingles based on the Jaccard similarity, or IoU.

        Parameters
        ----------
        vec : list
            List of Booleans for hashed k-shingles per text or length(shingles_all).
        vec_all : list
            List of multiple vec for each text, as defined above.

        Returns
        -------
        ious
            List of tuples of idx and iou. idx is the index of a text and the iou is the Jaccard similarity of a text
            with that text based on k-shingles.
        """
        ious = []
        vec_np = np.array(vec)

        for idx, vec_other in enumerate(vec_all):
            vec_other_np = np.array(vec_other)
            intersection = np.logical_and(vec_np, vec_other_np).sum()
            union = np.logical_or(vec_np, vec_other_np).sum()
            iou = intersection / union
            ious.append((idx, iou))

        return ious

    def __call__(self, k_shingles_rdd, k_shingles_all_rdd):
        """Compare hashed k-shingles per text based on teh Jaccard similarity.

        Parameters
        ----------
        k_shingles_rdd : pyspark.rdd.PipelinedRDD
            A list of (hashed) k-shingles for each text.
        k_shingles_all_rdd : pyspark.rdd.PipelinedRDD
            A list of all of the (hashed) k-shingles across all of the texts.

        Returns
        -------
        js_rdd : pyspark.rdd.PipelinedRDD
            A list of lists of tuples of (idx, iou) where for each text, its list contains the tuples of document
            indices and the text's similarity with the text of that document. Note documents are compared with
            themselves too yielding 1.0 Jaccard similarity.
        E.g.: The similarity measures of the document at index 11 with other documents.
        >>  js_rdd.collect()[11]
        [(0, 0.0019347037484885128),
         (1, 0.0011418783899514702),
         (2, 0.004890544946436889),
         (3, 0.0046443412368614035),
         (4, 0.010502625656414103),
         (5, 0.0),
         (6, 0.0018542555164101613),
         (7, 0.0025614754098360654),
         (8, 0.0067026624464717926),
         (9, 0.004962779156327543),
         (10, 0.0019553072625698325),
         (11, 1.0),
         (12, 0.002153316106804479),
         (13, 0.003216726980297547),
         (14, 0.003205128205128205),
         (15, 0.002745367192862045),
         (16, 0.005393478793822015),
         (17, 0.0027685492801771874),
         (18, 0.026589242053789732),
         (19, 0.0076937511728279225)]
        """
        # collect to list
        k_shingles_all_rdd_collected = k_shingles_all_rdd.collect()
        # Boolean vectorize the hashed k-shingles per doc
        k_shingles_onehot_rdd = \
            k_shingles_rdd.map(lambda x: self.vectorize(x, shingles_all=k_shingles_all_rdd_collected))
        # collect to list
        vec_all = k_shingles_onehot_rdd.collect()
        # get jaccard similarities
        js_rdd = k_shingles_onehot_rdd.map(lambda x: self.compare_sets(x, vec_all=vec_all))

        return js_rdd

