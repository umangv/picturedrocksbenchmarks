#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd

if __name__ == "__main__":
    for dataset in ["paul", "zeisel", "green", "zheng"]:
        interactions = np.load(f"../../output/{dataset}_interactions.npz")[
            "interaction"
        ]
        assert np.allclose(interactions.T, interactions)
        data = interactions[np.triu_indices(interactions.shape[0])]
        qs = np.quantile(data, np.linspace(0, 1, 20 + 1, True))
        h = np.histogram(data, qs)
        width = [(h[1][i + 1] - h[1][i]) for i in range(len(h[0]))]
        y = h[0] / width
        print(dataset)
        df = pd.DataFrame(dict(x=h[1], y=y.round().astype(int).tolist() + [0]))
        df = df.set_index("x")
        df.to_csv(f"{dataset}_histogram.csv")
