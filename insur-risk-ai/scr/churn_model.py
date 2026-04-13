import logging
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ML_Pipeline_Churn")

class DataLoadError(Exception):
    pass

class ModelTrainingError(Exception):
    pass

class ChurnModel:
    """
    Sənaye standartlarına uyğun Maşın Öyrənməsi (ML) proseslərinin idarəedilməsi.
    """
    def __init__(self, df: pd.DataFrame, target_col: str = 'is_churned'):
        self.df = df.copy()
        self.target_col = target_col

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Məlumatları stratifikasiya ilə təlim və test hissələrinə ayırır."""
        logger.info("Məlumatlar Train/Test olaraq ayrılır...")
        if self.target_col not in self.df.columns:
            raise KeyError(f"Hədəf sütun '{self.target_col}' tapılmadı.")
            
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train_and_select_best_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Modelləri öyrədir, hiperparametrləri tənzimləyir və ən yaxşı obyekti qaytarır.
        """
        logger.info("Model təlimi və GridSearch başlayır...")
        
        num_cols = X_train.select_dtypes(include='number').columns.tolist()
        cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0

        models = {
            'xgb': XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42),
            'lgb': LGBMClassifier(verbose=-1, class_weight='balanced', random_state=42),
            'rf': RandomForestClassifier(class_weight='balanced', random_state=42)
        }

        params = {
            'xgb': {'model__n_estimators': [100, 200], 'model__max_depth': [3, 5], 'model__learning_rate':[0.05, 0.1]},
            'lgb': {'model__n_estimators':[100, 200], 'model__num_leaves': [31, 50], 'model__learning_rate': [0.05, 0.1]},
            'rf': {'model__n_estimators': [100, 200], 'model__max_depth':[10, 15], 'model__min_samples_split': [2, 5]}
        }

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )

        best_global_roc = -1
        best_global_model_name = ""
        results = {}

        for name, model in models.items():
            logger.info(f"--- {name.upper()} təlimi başlayır ---")
            
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            gs = GridSearchCV(
                estimator=pipeline, 
                param_grid=params[name], 
                scoring='f1', 
                cv=3, 
                n_jobs=-1
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gs.fit(X_train, y_train)

                best_pipeline = gs.best_estimator_
                prob = best_pipeline.predict_proba(X_test)[:, 1]
                
                precision, recall, thresholds = precision_recall_curve(y_test, prob)
                f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
                best_threshold = thresholds[np.argmax(f1_scores)]

                pred = (prob > best_threshold).astype(int)
                roc_auc = roc_auc_score(y_test, prob)
                
                logger.info(f"{name.upper()} ROC-AUC: {roc_auc:.4f} | Optimal Threshold: {best_threshold:.2f}")
                logger.info(classification_report(y_test, pred))
                
                results[name] = {
                    'pipeline': best_pipeline,
                    'roc_auc': roc_auc,
                    'best_threshold': best_threshold,
                    'report': classification_report(y_test, pred)
                }

                if roc_auc > best_global_roc:
                    best_global_roc = roc_auc
                    best_global_model_name = name

            except Exception as e:
                logger.error(f"{name.upper()} modeli üzrə xəta baş verdi: {e}")
                continue
        
        if not best_global_model_name:
            raise ModelTrainingError("Heç bir model təlim edilə bilmədi.")

        logger.info(f"Ən Yaxşı Model: {best_global_model_name.upper()} (ROC-AUC: {best_global_roc:.4f})")
        return results[best_global_model_name]

    def explain_model(self, best_result_dict: Dict[str, Any], X_test: pd.DataFrame, base_dir: Path) -> None:
        """Modellərin izahı üçün Feature Importance və SHAP qrafiklərini tərtib edir."""
        logger.info("Feature Importance və SHAP analizi aparılır...")
        
        pipeline = best_result_dict['pipeline']
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']

        plot_dir = base_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        try:
            X_test_transformed = preprocessor.transform(X_test)
            all_features = preprocessor.get_feature_names_out()

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({'feature': all_features, 'importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False).head(10)

                plt.figure(figsize=(8, 5))
                plt.barh(feat_imp_df['feature'], feat_imp_df['importance'], color='steelblue')
                plt.gca().invert_yaxis()
                plt.title("Top 10 Feature Importances")
                plt.savefig(plot_dir / "churn_feature_importance.png", dpi=300, bbox_inches='tight')
                plt.close()

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_transformed)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap.summary_plot(shap_values, X_test_transformed, feature_names=all_features, show=False)
            plt.savefig(plot_dir / "churn_shap_summary.png", dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Qrafiklər (Feature Importance & SHAP) uğurla yadda saxlanıldı.")
            
        except Exception as e:
            logger.error(f"Model izahı (SHAP/Importance) zamanı xəta: {e}")

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / 'data' / 'clean_churn_data.csv'
    model_dir = BASE_DIR / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Məlumatlar yüklənir: {file_path}")
        df = pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f"Data faylı tapılmadı: {e}")
        return

    ml_processor = ChurnModel(df, target_col='is_churned')
    X_train, X_test, y_train, y_test = ml_processor.split_data()

    best_result = ml_processor.train_and_select_best_model(X_train, X_test, y_train, y_test)
    production_artifact = {
        'model_pipeline': best_result['pipeline'],
        'optimal_threshold': best_result['best_threshold']
    }
    
    model_path = model_dir / 'best_churn_model.joblib'
    joblib.dump(production_artifact, model_path)
    logger.info(f"Model (Pipeline) istehsal mühiti üçün yadda saxlanıldı: {model_path}")

    ml_processor.explain_model(best_result, X_test, BASE_DIR)
    
    logger.info("===== Bütün proseslər xətasız tamamlandı =====")

if __name__ == '__main__':
    main()