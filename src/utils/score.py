import numpy as np
import pandas as pd


def rescale_score(pd, base_odds=120, pdo=100, base_point=500, ndigit=2):
    odds = (1 - pd) / pd
    odds = np.where(pd == 0, np.inf, np.where(
        pd == 1, 0, odds))  # Handle pd = 0 or 1
    factor = pdo / np.log(2)
    offset = base_point - factor * np.log(base_odds)
    score = np.round(offset + factor * np.log(odds), ndigit)
    return score
