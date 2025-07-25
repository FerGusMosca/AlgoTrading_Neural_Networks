from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class PCACalcualtor():

    @staticmethod
    def  calculate_PCA(pivot_df,output_symbol):
        scaled_data = StandardScaler().fit_transform(pivot_df.drop(columns=["date"]))
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(scaled_data)

        pivot_df[output_symbol] = principal_component[:, 0]

        return pivot_df