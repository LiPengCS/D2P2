from TFs.normalizer import Normalizer
from TFs.outlier_cleaner import OutlierCleaner
from TFs.mv_imputer import NumMVImputer, NumCatMVImputer, CatMVImputer
from TFs.feature_selector import FeatureSelector
from TFs.discretizer import Discretizer
from TFs.identity import Identity
from TFs.nonlinear_transformer import NonlinearTransformer

space = [
    {
        "name": "missing_value_imputation",
        "num_tf_options": [NumMVImputer("mean"),
                           NumMVImputer("median"),
                           NumMVImputer("DT"),
                           NumMVImputer("KNN"),
                           NumMVImputer("MICE"),
                           NumCatMVImputer("mode")],
        "cat_tf_options": [
            NumCatMVImputer("mode"),
            CatMVImputer("dummy"),
        ],
        "default": [NumMVImputer("mean"), NumCatMVImputer("mode")],
        "init": [(NumMVImputer("mean"), 0.5), (NumCatMVImputer("mode"), 0.5)]
    },
    {
        "name": "normalization",
        "tf_options": [Normalizer("ZS"),
                       Normalizer("MM"),
                       Normalizer("MA"),
                       Normalizer("RS")],
        "default": Normalizer("ZS"),
        "init": (Normalizer("ZS"), 0.5)
    },
    # {
    #     "name": "nonlinear_transformation",
    #     "tf_options": [NonlinearTransformer("UQT"),
    #                    NonlinearTransformer("NQT"),
    #                    NonlinearTransformer("Power"),
    #                    Identity()],
    #     "default": Identity(),
    #     "init": (Identity(), 0.5)
    # },
    {
        "name": "cleaning_outliers",
        "tf_options": [OutlierCleaner("ZS"),
                       OutlierCleaner("MAD"),
                       OutlierCleaner("IQR"),
                       Identity()],
        "default": Identity(),
        "init": (Identity(), 0.5)
    },
    {
      "name": "discretization",
      "tf_options": [Discretizer(n_bins=5, strategy="uniform"),
                     Discretizer(n_bins=5, strategy="quantile"),
                     Identity()],
      "default": Identity(),
      "init": (Identity(), 0.5)
    }
]
