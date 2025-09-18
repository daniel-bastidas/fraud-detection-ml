import pandas as pd
from src.data_prep import build_preprocess_pipeline

def test_build_preprocess_pipeline():
    pre = build_preprocess_pipeline(num_cols=["amount"], cat_cols=["country"])
    df = pd.DataFrame({"amount":[10,20], "country":["AR","BR"]})
    Xt = pre.fit_transform(df)
    assert Xt.shape[0] == 2