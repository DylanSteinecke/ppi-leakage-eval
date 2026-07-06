import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


class BM25Vectorizer:
    """
    Minimal BM25 vectorizer for protein k-mers.
    API matches sklearn vectorizers: fit(), transform().
    """

    def __init__(self, k: int = 3, k1: float = 1.5, b: float = 0.75):
        self.k = k
        self.k1 = k1
        self.b = b
        self.count_vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            lowercase=False,
        )
        self.idf_ = None
        self.avgdl_ = None


    def fit(self, sequences):
        counts = self.count_vectorizer.fit_transform(sequences).tocsr()
        n_docs = counts.shape[0]

        # document frequency per k-mer
        df = np.diff(counts.tocsc().indptr)

        self.idf_ = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self.avgdl_ = float(
            np.asarray(counts.sum(axis=1)).ravel().mean())
        self.avgdl_ = max(self.avgdl_, 1e-12)

        return self


    def transform(self, sequences):
        if (self.idf_ is None) or (self.avgdl_ is None):
            raise RuntimeError(
                "BM25Vectorizer must be fit before transform.")

        counts = self.count_vectorizer.transform(sequences).tocsr().astype(np.float64)
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel()

        coo = counts.tocoo()
        tf = coo.data
        rows = coo.row
        cols = coo.col

        denom = tf + self.k1 * (
            1.0 - self.b + self.b * doc_lengths[rows] / self.avgdl_
        )

        data = self.idf_[cols] * ((tf * (self.k1 + 1.0)) / denom)

        return csr_matrix((data, (rows, cols)), shape=counts.shape)
