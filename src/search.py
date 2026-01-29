import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class ICD10Searcher:
    def __init__(self, xlsx_path: str):
        self.df = pd.read_excel(xlsx_path)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.matrix = self.vectorizer.fit_transform(self.df["Описание"])

    def search(self, query: str, top_k: int = 1):
        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "code": self.df.iloc[idx]["Код"],
                "description": self.df.iloc[idx]["Описание"],
                "score": float(scores[idx])
            })
        return results
