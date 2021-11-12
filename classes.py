import numpy as np
from sympy.ntheory.generate import nextprime
import time


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


class Vectorizer():
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

    def __call__(self, k_shingles_rdd, k_shingles_all_rdd):
        # collect to list
        k_shingles_all_rdd_collected = k_shingles_all_rdd.collect()
        # Boolean vectorize the hashed k-shingles per doc
        k_shingles_vectorized_rdd = \
            k_shingles_rdd.map(lambda x: self.vectorize(x, shingles_all=k_shingles_all_rdd_collected))

        return k_shingles_vectorized_rdd


class CompareSets():
    """A class CompareSets that computes the Jaccard similarity of two sets of integers – two sets
    of hashed shingles."""
    def __init__(self, ):
        pass

    def compare_sets(self, vec, vec_all):
        """Compare vectorized sets of k-shingles (Boolean vector, 1 if shingle in text, 0 otherwise) based on the
        Jaccard similarity, or IoU.

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

    def __call__(self, k_shingles_vectorized_rdd):
        """Compare Boolean vectors of (hashed) k-shingles per text based on the Jaccard similarity.

        Parameters
        ----------
        k_shingles_vectorized_rdd : pyspark.rdd.PipelinedRDD
            A list of Booleans for (hashed) k-shingles from across all texts for each text. 1 if shingle (from all
            shingles) is present in text, 0 otherwise. All vectors are of the same length.

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
        vec_all = k_shingles_vectorized_rdd.collect()
        # get jaccard similarities
        js_rdd = k_shingles_vectorized_rdd.map(lambda x: self.compare_sets(x, vec_all=vec_all))

        return js_rdd


class MinHashing():
    """MinHashing uses the min-hashing algorithm to generate a n_signatures long signature given a vectorized
    hashed k-shingle of a text. The vectorized hashed k-shingle of a text is an n_shingles (=len(k_shingles_all_rdd))
    long Boolean vector that represents if a shingle is in the text (1s) or not (0s). n_shingles is the number of all
    distinct shingles across all text documents."""
    def __init__(self, n_signatures, prime_modulo, seed):
        """Init.

        Parameters
        ----------
        n_signatures : int
            The length of the min-hashed signatures.
        prime_modulo : int
            The prime modulo of the hashing function of the min-hashing algorithm of the form (ax+b)%c. It is the next
            prime after the integer representing the length of all of the distinct shingles across all text documents,
            n_shingles.
        seed : int
            Seed for reproducibility.
        """
        self.n_signatures = n_signatures
        self.prime_modulo = prime_modulo
        self.seed = seed

    def minhash_func(self, x, param):
        """Minhash hashing function of the form (ax+b)%c. Gives the signature of a hash function.

        Parameters
        ----------
        x : int
            The integer to be hashed. Here, an integer representing the row number of a k-shingle that is present in
            the text document from across all of the text documents.
        param : list
            List of integers representing a,b, and c of the minhashing hashing function.

        Returns
        -------
        int
            The hashed value of the integer x, the signature entry of a hash function.
        """
        # h(x) = (ax + b) % c
        a, b, c = param
        return (a * x + b) % c

    def minhash_funcs(self, xs, params):
        """Minhash hashing functions of the form (ax+b)%c. Gives the signatures of all of the hash functions as defined
        by their parameters in params. The signature length is self.n_signatures = len(params).

        Parameters
        ----------
        xs : list
            List of integers to be hashed. The list of row indices of k-shingles that are present in the text document
            from across all of the documents.
        params : list
            List of lists of integers representing a,b, and c of the n_signatures number of minhashing
            hashing functions.

        Returns
        -------
        int
            The hashed value of the integer x, the signature entry of a hash function.
        """
        signature = []

        for param in params:
            minhash_value_per_func = min([self.minhash_func(x, param) for x in xs])
            signature.append(minhash_value_per_func)

        return signature

    def minhash_param(self, seed):
        """The parameters of a minhash hashing function of the form (ax+b)%c.

        Parameters
        ----------
        seed : int
            Seed for reproducibility.

        Returns
        -------
        tuple
            a,b, and c of the minhash hashing function.

        """
        np.random.seed(seed)
        a = np.random.randint(1, self.prime_modulo)
        np.random.seed(seed + 123)
        b = np.random.randint(1, self.prime_modulo)
        return a, b, self.prime_modulo

    def minhash_params(self):
        """The parameters of a minhash hashing function of the form (ax+b)%c.

        Parameters
        ----------

        Returns
        -------
        list
            list of tuples of a,b, and c of the minhash hashing function.
        """
        np.random.seed(self.seed)
        seeds = np.random.randint(low=1, high=2 * self.n_signatures, size=self.n_signatures)
        return [self.minhash_param(seed=seeds[i]) for i in range(self.n_signatures)]

    def __call__(self, k_shingles_vectorized_rdd):
        """Computes the signatures by MinHashing of hashed k-shingles. The hashed k-shingles are represented as
        Boolean vectors of 1s if a shingle from across all the texts is in the text, 0s otherwise. The signatures
        are self.n_signatures long.

        Parameters
        ----------
        k_shingles_vectorized_rdd : pyspark.rdd.PipelinedRDD
            A list of Booleans for (hashed) k-shingles from across all texts for each text. 1 if shingle (from all
            shingles) is present in text, 0 otherwise. All vectors are of the same length.

        Returns
        -------
        signatures_rdd : pyspark.rdd.PipelinedRDD
            The n_signatures long MinHashing signatures of k-shingles per text.
        """
        params = self.minhash_params()
        k_shingles_vectorized_args_rdd = \
            k_shingles_vectorized_rdd.map(lambda x: list(np.argwhere(x).flatten()))
        signatures_rdd = \
            k_shingles_vectorized_args_rdd.map(lambda x: self.minhash_funcs(xs=x, params=params))
        return signatures_rdd


class CompareSignatures():
    """Compares MinHashed signatures of texts. The similarity is measured as the ratio of equal signatures to the
    number of all of the signatures."""
    def __init__(self):
        pass

    def compare_signatures(self, vec, vec_all):
        """Compare MinHashed signatures of texts.

        Parameters
        ----------
        vec : list
            A MinHashed signature represented as a list of integers.

        Returns
        -------
        vec_all : list
            List of lists of MinHashed signatures.
        """
        similarities = []
        vec_np = np.array(vec)

        for idx, vec_other in enumerate(vec_all):
            vec_other_np = np.array(vec_other)

            similarities.append((idx, (vec_np == vec_other_np).sum() / vec_np.size))

        return similarities

    def __call__(self, signatures_rdd):
        """Compare MinHashed signatures of texts.

        Parameters
        ----------
        signatures_rdd : pyspark.rdd.PipelinedRDD
            List of MinHashed signatures.

        Returns
        -------
        signature_similarities_rdd : pyspark.rdd.PipelinedRDD
            The similarities of signatures.
            E.g.: The similarity measures of the document at index 11 with other documents.
            >>  js_rdd.collect()[11]
            [(0, 0.02),
             (1, 0.0),
             (2, 0.01),
             (3, 0.0),
             (4, 0.03),
             (5, 0.0),
             (6, 0.0),
             (7, 0.0),
             (8, 0.04),
             (9, 0.01),
             (10, 0.0),
             (11, 1.0),
             (12, 0.0),
             (13, 0.0),
             (14, 0.0),
             (15, 0.0),
             (16, 0.01),
             (17, 0.01),
             (18, 0.04),
             (19, 0.03)]
            """
        signatures_rdd_collected = signatures_rdd.collect()
        signature_similarities_rdd = \
            signatures_rdd.map(lambda x: self.compare_signatures(vec=x, vec_all=signatures_rdd_collected))

        return signature_similarities_rdd


class LSH():
    """A class for Locally Sensitive Hashing (LSH) that implements the LSH technique: given a collection of minhash
    signatures (integer vectors) and a similarity threshold t, the LSH class (using banding and hashing) finds
    candidate pairs of signatures agreeing on at least fraction t of their components."""
    def __init__(self, n_signatures, n_bands, n_buckets, hash_to_n_min=1):
        self.n_signatures = n_signatures
        self.n_bands = n_bands
        self.n_buckets = n_buckets
        self.n_rows_per_band = int(n_signatures / n_bands)
        self.sim_thresh = (1 / n_bands) ** (1 / self.n_rows_per_band)
        self.hash_to_n_min = hash_to_n_min

    def lsh_hash(self, rows_in_band):
        """ Hash the rows of a column in a MinHash signature vector.

        Parameters
        ----------
        rows_in_band : list
            The list of MinHash signature integers representing rows in a band of a column of a signature.

        Returns
        -------
        int
            The hashed rows per band of a column in a signature.
        """
        # type cast rows_in_band list to tuple so that it is immutable and hashable
        # moduleo is n_buckets so biggest hash is the integer n_buckets (hashing into buckets)
        return hash(tuple(rows_in_band)) % self.n_buckets

    def lsh(self, vec):
        """ LSH algorithm. For a given signature vector, for all bands, hash the rows in the band. Returns a list of
        hash values of length n_bands. Each hash value is the hashed value of the rows of each band in the signature
        vector. The number of rows is n_rows_per_band. The list of rows are type casted to tuples to make the
        immutable and hence hashable.

        Parameters
        ----------
        vec : list
            A MinHash signature vector.

        Returns
        -------
        list
            The LSHed vector of a signature vector. The LSHed vector is of length n_bands
        """
        return [self.lsh_hash(rows_in_band=vec[self.n_rows_per_band * b: self.n_rows_per_band * (b + 1)])
                for b in range(self.n_bands)]

    def candidates(self, vec, vec_all):
        """ Return text document candidates that are similar. The candidates are represented as list of tuples where
        the first element in the tuple is the idx of a text and the second element is a list of the indices of similar
        documents. Similarity is defined as two MinHash signatures (of texts) hashing to the same bucket for any band
        in the LSH algorithm equal to or more time than self.hash_to_n_min.

        Parameters
        ----------
        vec : list
            An LSHed vector of length n_bands representing a text doc.
        vec_all : list
            List of LSHed vectors of length n_bands representing text docs.

        Returns
        -------
        list
            A list of integers where each integer is the index of a text doc similar to the text doc represented by vec,
            where similarity is defined by LSH.
        """
        candidate_pairs = []
        vec_np = np.array(vec)

        for idx, vec_other in enumerate(vec_all):
            vec_other_np = np.array(vec_other)

            hash_to_n = (vec_np == vec_other_np).sum()

            if self.hash_to_n_min <= hash_to_n:
                candidate_pairs.append(idx)

        return candidate_pairs

    def __call__(self, signatures_rdd):
        """ Generate similar candidates of texts (represented by MinHashed signatures) based on LSH. The similarity
        threshold is self.sim_thresh.

        Parameters
        ----------
        signatures_rdd : pyspark.rdd.PipelinedRDD
            The MinHashed signatures of texts.

        Returns
        -------
        candidates_rdd : pyspark.rdd.PipelinedRDD
            List of tuples where the first element of the tuple is the index of a text doc, and the second element is
            a list of indices of similar candidate documents based on LSH.
        """
        lsh_rdd = signatures_rdd.map(lambda x: self.lsh(vec=x))
        lsh_rdd_collected = lsh_rdd.collect()
        candidates_rdd = \
            lsh_rdd.map(lambda x: self.candidates(vec=x, vec_all=lsh_rdd_collected)).zipWithIndex().\
            map(lambda x: (x[1], x[0]))

        return candidates_rdd

    def __repr__(self):
        repr_str = \
            f"LSH:\n" \
            f"n_signatures={self.n_signatures}\n" \
            f"n_bands={self.n_bands}\n" \
            f"n_buckets={self.n_buckets}\n" \
            f"n_rows_per_band={self.n_rows_per_band}\n" \
            f"sim_thresh={self.sim_thresh}\n" \
            f"hash_to_n_min={self.hash_to_n_min}\n"

        return repr_str


class FindTextuallySimilarDocuments():
    def __init__(self, k, n_signatures, prime_modulo, seed, n_bands, n_buckets, hash_to_n_min=1):
        """Init."""
        self.k = k
        self.n_signatures = n_signatures
        self.prime_modulo = prime_modulo
        self.seed = seed
        self.n_bands = n_bands
        self.n_buckets = n_buckets
        self.hash_to_n_min = hash_to_n_min

    def __call__(self, df_data):
        """Run full algorithm. """
        start = time.time()

        # make texts rdd
        texts_rdd = df_data.select("text").rdd.flatMap(lambda x: x)

        # shingling
        shingling = Shingling(k=self.k)
        k_shingles_rdd, k_shingles_all_rdd = shingling(texts_rdd=texts_rdd)
        n_shingles = k_shingles_all_rdd.count()
        print(f"Found n_shingles={n_shingles} distinct shingles across all docs with k={self.k}")
        prime_modulo = nextprime(n_shingles)

        # vectorize for shingling comparison and minhashing
        vectorizer = Vectorizer()
        k_shingles_vectorized_rdd = \
            vectorizer(k_shingles_rdd=k_shingles_rdd, k_shingles_all_rdd=k_shingles_all_rdd)

        # compare k-shingles
        compare_sets = CompareSets()
        js_rdd = compare_sets(k_shingles_vectorized_rdd=k_shingles_vectorized_rdd)

        # minhashing
        min_hashing = MinHashing(n_signatures=self.n_signatures, prime_modulo=self.prime_modulo, seed=self.seed)
        signatures_rdd = min_hashing(k_shingles_vectorized_rdd=k_shingles_vectorized_rdd)

        # compare minhashed signatures
        compare_signatures = CompareSignatures()
        signature_similarities_rdd = compare_signatures(signatures_rdd=signatures_rdd)

        # lsh
        lsh = LSH(n_signatures=self.n_signatures, n_bands=self.n_bands, n_buckets=self.n_buckets)
        print(lsh)
        candidates_rdd = lsh(signatures_rdd=signatures_rdd)

        end = time.time()
        print(f"Execution time: {end - start} seconds")
