from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class PCACalcualtor():

    @staticmethod
    def calculate_PCA(pivot_df, output_symbol, excluded_columns=None):
        if excluded_columns is None:
            excluded_columns = []

        # Separate columns to transform and columns to exclude
        cols_to_transform = [col for col in pivot_df.columns if col not in excluded_columns + ["date"]]

        # Rename columns with 'feat_' prefix
        renamed = {col: f"feat_{col}" for col in cols_to_transform}
        pivot_df = pivot_df.rename(columns=renamed)

        # Build the feature matrix
        feature_cols = list(renamed.values())
        scaled_data = StandardScaler().fit_transform(pivot_df[feature_cols])

        # Run PCA
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(scaled_data)

        # Flip sign if needed (make more stress = higher values)
        if principal_component.mean() < 0:
            principal_component *= -1

        # Add PCA result to DataFrame
        pivot_df[output_symbol] = principal_component[:, 0]

        return pivot_df
