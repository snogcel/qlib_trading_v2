from qlib.model.base import Model
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd

class QuantileLGBModel(LGBModel):
    def __init__(self, alpha=0.9, **kwargs):
        super().__init__(loss="mse", **kwargs)  # temporary placeholder for base class
        self.params["objective"] = "quantile"
        self.params["alpha"] = alpha


class MultiQuantileModel(Model):
    def __init__(self, quantiles=[0.1, 0.5, 0.9], lgb_params=None):
        self.quantiles = quantiles
        self.models = {}
        self.evals_result_dict = {}
        self.lgb_params = lgb_params or {
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": 6,
            "n_estimators": 1000
        }

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=None,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        **kwargs,
    ):        
        """ df_train = dataset.prepare(segments=["train"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        X, y = df_train["feature"], df_train["label"] """
       
        for q in self.quantiles:
            evals_result = {}
            model = QuantileLGBModel(alpha=q, **self.lgb_params[q])
            print(self.lgb_params[q])
            model.fit(dataset, num_boost_round, early_stopping_rounds, verbose_eval, evals_result, reweighter, **kwargs)
            self.models[q] = model 
            self.evals_result_dict[q] = evals_result          

    def predict(self, dataset, segment):
        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(dataset, segment).rename(f"quantile_{q:.2f}")
        return pd.concat(preds.values(), axis=1)
    
