import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Feature_Engineering")

class DataSaveError(Exception): pass

class FeatureEngineering:
    """Churn və Fraud modelləri üçün ayrı-ayrı əlamət generasiyası edən sinif."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _apply_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Hər iki model üçün ortaq olan təməl riyazi əməliyyatlar."""
        df = data.copy()
        df = df.drop(columns=['customer_id', 'policy_id'], errors='ignore')
        
        df['has_claim'] = (df['num_claims'] > 0).astype(int)
        df['claim_freq'] = np.where(df['tenure_months'] > 0, df['num_claims'] / df['tenure_months'], 0)
        df['monthly_premium'] = np.where(df['tenure_months'] > 0, df['premium_amount'] / df['tenure_months'], 0)
        df['age_policy'] = df['age'] * df['premium_amount']
        df['tenure_claim'] = df['tenure_months'] * df['num_claims']
        
        return df

    def get_churn_data(self) -> pd.DataFrame:
        """Churn (Müştəri İtkisi) Modeli üçün xüsusi feature-lar."""
        logger.info("Churn modeli üçün Feature Engineering tətbiq edilir...")
        try:
            df = self._apply_base_features(self.df)
            
            df['settlement_ratio'] = np.where(df['total_claim_amount'] > 0, df['total_claim_amount_settled'] / df['total_claim_amount'], 0)
            df['profitability'] = df['premium_amount'] - df['total_claim_amount_settled']
            df['suspicious'] = ((df['total_claim_amount'] / df['premium_amount'] > 2) & (df['avg_settlement_time'] > 20)).astype(int)
            df['fast_settlement'] = (df['avg_settlement_time'] <= 7).astype(int)
            df['delay_risk'] = df['avg_settlement_time'] * df['num_claims']
            df['rejection_ratio'] = np.where(df['num_claims'] > 0, df['is_rejected'] / df['num_claims'], 0)
            df['claim_to_premium'] = np.where(df['premium_amount'] > 0, df['total_claim_amount'] / df['premium_amount'], 0)
            df['claim_severity'] = np.where(df['num_claims'] > 0, df['total_claim_amount'] / df['num_claims'], 0)
            
            return df.drop(columns=['is_fraudulent'], errors='ignore')
        except KeyError as e:
            logger.error(f"Churn Feature Engineering zamanı xəta: {e}")
            raise

    def get_fraud_data(self) -> pd.DataFrame:
        """Fraud (Dələduzluq) Modeli üçün xüsusi feature-lar (Data Leakage olmadan)."""
        logger.info("Fraud modeli üçün Feature Engineering tətbiq edilir...")
        try:
            df = self._apply_base_features(self.df)
            
            df['claim_to_premium'] = np.where(df['premium_amount'] > 0, df['total_claim_amount'] / df['premium_amount'], 0)
            df['claim_severity'] = np.where(df['num_claims'] > 0, df['total_claim_amount'] / df['num_claims'], 0)
            
            leakage_cols =['total_claim_amount_settled', 'avg_settlement_time', 'is_rejected', 'is_churned']
            df = df.drop(columns=leakage_cols, errors='ignore')
            
            return df
        except KeyError as e:
            logger.error(f"Fraud Feature Engineering zamanı xəta: {e}")
            raise

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / 'data'
    
    merged_data_path = data_dir / 'merged_data.csv'
    churn_data_path = data_dir / 'clean_churn_data.csv'
    fraud_data_path = data_dir / 'clean_fraud_data.csv'

    try:
        df = pd.read_csv(merged_data_path)
    except FileNotFoundError as e:
        logger.error(f"Fayl tapılmadı: {e}")
        return

    fe = FeatureEngineering(df)
    
    churn_df = fe.get_churn_data()
    fraud_df = fe.get_fraud_data()

    try:
        churn_df.to_csv(churn_data_path, index=False)
        fraud_df.to_csv(fraud_data_path, index=False)
        logger.info("Həm Churn, həm də Fraud məlumatları uğurla yadda saxlanıldı.")
    except Exception as e:
        raise DataSaveError("Fayllar saxlanıla bilmədi.") from e

if __name__ == '__main__':
    main()