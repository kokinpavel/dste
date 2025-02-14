import pandas as pd


class PreprocessModel:
    def __init__(
        self,
        cols_to_bin: list = ["age", "duration"],
        bins: int = 20,
        min_count: int = 20,
        fill_value: int = 999,
    ):
        self.cols_to_bin = cols_to_bin
        self.bins = bins
        self.min_count = min_count
        self.fill_value = fill_value
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        self.get_campaign_to_save(df)
        self.get_bins(df)
        self.is_fitted = True

    def transform(self, df: pd.DataFrame):
        # Small check
        assert self.is_fitted, "The model is not fitted yet!"

        # Pipeline
        self.target_encoding(df)
        self.bin_columns(df)
        self.correct_education(df)
        self.fill_campaign(df)
        return df

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    def target_encoding(self, df: pd.DataFrame):
        # Target to binery
        df["y"] = df["y"].map({"no": 0, "yes": 1})

    def bin_columns(self, df: pd.DataFrame):
        # Bin
        for col in self.cols_to_bin:
            df[f"{col}_bins"] = pd.qcut(df[col], q=self.bins).astype(str)

    def correct_education(self, df: pd.DataFrame):
        # Correct education
        df["education"] = df["education"].apply(
            lambda x: "basic" if "basic" in x else x
        )

    def get_campaign_to_save(self, df: pd.DataFrame):
        # Determine campaigns with less than min_count
        self.campaign_to_save = (
            df["campaign"]
            .value_counts()[df["campaign"].value_counts() > self.min_count]
            .index
        )

    def fill_campaign(self, df: pd.DataFrame):
        # Replace with fill_value
        df.loc[~df["campaign"].isin(self.campaign_to_save), "campaign"] = (
            self.fill_value
        )

    def get_bins(self, df: pd.DataFrame):
        # Collect bin edges
        self.bin_edges = {}
        for col in self.cols_to_bin:
            _, edges = pd.qcut(df[col], q=self.bins, retbins=True, duplicates="drop")
            self.bin_edges[col] = edges

    def fill_bins(self, df: pd.DataFrame):
        # Small check
        assert self.cols_to_bin == self.bin_edges, (
            "The model is not fitted for some cols!"
        )

        # Bin cols
        for col in self.cols_to_bin:
            df[f"{col}_bins"] = pd.cut(df[col], bins=self.bin_edges[col]).astype(str)
