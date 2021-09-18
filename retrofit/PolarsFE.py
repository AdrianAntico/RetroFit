# Module: PolarsFE
# Author: Adrian Antico <adrianantico@gmail.com>
# License: MIT
# Release: retrofit 0.1.4
# Last modified : 2021-09-15

import polars as pl
from retrofit.FeatureEngineering import FeatureEngineering


class FE(FeatureEngineering):
    def __init__(self) -> None:
        super().__init__()
