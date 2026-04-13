import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EDA_Processor")

class DataLoadError(Exception): pass
class DataSaveError(Exception): pass

class InsuranceEDA:
    """Xam sığorta məlumatlarının kəşfiyyat analizi və birləşdirilməsi."""
    def __init__(self, policies_df: pd.DataFrame, claims_df: pd.DataFrame):
        self.policies_df = policies_df.copy()
        self.claims_df = claims_df.copy()

    def explore_data(self) -> None:
        logger.info("EDA prosesi başlayır...")
        datasets = {"Policies Data": self.policies_df, "Claims Data": self.claims_df}
        for name, df in datasets.items():
            logger.info(f"\n{'='*40}\n{name}\n{'='*40}")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
            logger.info(f"Duplicated Rows: {df.duplicated().sum()}")

    def merge_and_aggregate(self) -> pd.DataFrame:
        logger.info("Məlumatlar aqreqasiya edilir və birləşdirilir...")
        try:
            self.claims_df['is_rejected'] = (self.claims_df['claim_status'] == 'Rejected').astype(int)

            claims_agg = self.claims_df.groupby('policy_id').agg({
                'claim_amount_requested': 'sum',
                'claim_amount_settled': 'sum',
                'settlement_time_days': 'mean',
                'policy_id': 'size',
                'is_fraudulent': 'max',
                'is_rejected': 'sum'
            }).rename(columns={
                'claim_amount_requested': 'total_claim_amount',
                'claim_amount_settled': 'total_claim_amount_settled',
                'settlement_time_days': 'avg_settlement_time',
                'policy_id': 'num_claims'
            }).reset_index()
            
            final_df = self.policies_df.merge(claims_agg, on='policy_id', how='left')
            
            claim_cols =['total_claim_amount', 'total_claim_amount_settled', 'avg_settlement_time', 'num_claims', 'is_fraudulent', 'is_rejected']
            final_df[claim_cols] = final_df[claim_cols].fillna(0)
            
            return final_df
        except Exception as e:
            logger.error(f"Birləşdirmə zamanı xəta: {e}")
            raise

    def calculate_kpis(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        try:
            total_premium = df['premium_amount'].sum()
            loss_ratio = (df['total_claim_amount_settled'].sum() / total_premium * 100) if total_premium else 0.0
            
            unique_policies = df['policy_id'].nunique()
            churn_rate = (df['is_churned'].sum() / unique_policies * 100) if unique_policies else 0.0
            fraud_rate = (df['is_fraudulent'].sum() / unique_policies * 100) if unique_policies else 0.0

            logger.info(f"KPI - Loss Ratio: {loss_ratio:.2f}% | Churn Rate: {churn_rate:.2f}% | Fraud Rate: {fraud_rate:.2f}%")
            return loss_ratio, churn_rate, fraud_rate
        except KeyError as e:
            logger.error(f"KPI hesablanması üçün tələb olunan sütun tapılmadı: {e}")
            raise

    def generate_plots(self, df: pd.DataFrame, base_dir: Path) -> None:
        logger.info("Qrafiklər yaradılır...")
        plot_dir = base_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_df = df.copy()
        plot_df['bins_tenure_months'] = pd.cut(plot_df['tenure_months'], bins=[0, 3, 6, 9, 12, 18, 24, 36, 48, 72, 96, 120])
        
        sns.set_style('whitegrid')
        plt.figure(figsize=(10, 5))
        sns.barplot(x='bins_tenure_months', y='is_churned', data=plot_df)
        plt.title('Churn Rate by Tenure')
        plt.xticks(rotation=45)
        plt.savefig(plot_dir / "churn_rate_by_tenure.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Xam məlumatlar yüklənir...")
        df_p = pd.read_csv(data_dir / 'policies_crm.csv')
        df_c = pd.read_csv(data_dir / 'claims_data.csv')
    except FileNotFoundError as e:
        logger.error(f"Fayl tapılmadı: {e}")
        return

    eda = InsuranceEDA(df_p, df_c)
    eda.explore_data()
    merged_df = eda.merge_and_aggregate()
    eda.calculate_kpis(merged_df)
    eda.generate_plots(merged_df, BASE_DIR)

    logger.info(merged_df.info())

    merged_data_path = data_dir / 'merged_data.csv'
    try:
        merged_df.to_csv(merged_data_path, index=False)
        logger.info(f"Birləşdirilmiş data saxlanıldı: {merged_data_path}")
    except Exception as e:
        raise DataSaveError("Fayl saxlanıla bilmədi.") from e

if __name__ == '__main__':
    main()