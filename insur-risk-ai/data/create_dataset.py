import pandas as pd
import numpy as np
import os
from pathlib import Path

# Təkrarlana bilən nəticələr üçün random seed
np.random.seed(42)

# Sətir sayı
N_POLICIES = 4500

# -------------------------------------------------------------
# 1. POLICIES_CRM CƏDVƏLİNİN YARADILMASI
# -------------------------------------------------------------

# Unikal ID-lər
policy_ids =[f"POL-{i}" for i in range(10001, 10001 + N_POLICIES)]
customer_ids =[f"CUST-{i}" for i in range(10001, 10001 + N_POLICIES)]

age = np.random.randint(18, 76, size=N_POLICIES)
gender = np.random.choice(['Male', 'Female'], size=N_POLICIES)
policy_type = np.random.choice(['Motor', 'Health', 'Property'], size=N_POLICIES, p=[0.45, 0.35, 0.20])
premium_amount = np.round(np.random.uniform(100, 2500, size=N_POLICIES), 2)
sales_channel = np.random.choice(['Broker', 'Direct', 'Online', 'Bank'], size=N_POLICIES, p=[0.3, 0.2, 0.3, 0.2])
tenure_months = np.random.randint(1, 121, size=N_POLICIES)

# İS_CHURNED Məntiqi Qaydası:
# Gənclər (<30), yeni müştərilər (<12 ay) və ya Online kanaldan gələnlər üçün tərk etmə (churn) ehtimalı artır.
base_churn_prob = 0.10 # Baza ehtimal
churn_prob = base_churn_prob + \
             (age < 30) * 0.25 + \
             (tenure_months < 12) * 0.30 + \
             (sales_channel == 'Online') * 0.15

# Ehtimalı 0.05 ilə 0.95 arasında məhdudlaşdırırıq
churn_prob = np.clip(churn_prob, 0.05, 0.95)

# Binomial paylanma ilə is_churned dəyərlərinin (0 və 1) təyini
is_churned = np.random.binomial(1, churn_prob)

# DataFrame yaradılması
policies_df = pd.DataFrame({
    'customer_id': customer_ids,
    'age': age,
    'gender': gender,
    'policy_id': policy_ids,
    'policy_type': policy_type,
    'premium_amount': premium_amount,
    'sales_channel': sales_channel,
    'tenure_months': tenure_months,
    'is_churned': is_churned
})

# -------------------------------------------------------------
# 2. CLAIMS_DATA CƏDVƏLİNİN YARADILMASI
# -------------------------------------------------------------

claims_data =[]
claim_counter = 10001

for pid in policy_ids:
    # Hər polisin neçə şikayəti ola bilər? (Bəzilərində 0, bəzilərində 1-dən çox)
    # Təxminən ümumi datanın 3500-4000 civarında olması üçün ehtimallar:
    num_claims = np.random.choice([0, 1, 2, 3], p=[0.35, 0.45, 0.15, 0.05])
    
    for _ in range(num_claims):
        claim_id = f"CLM-{claim_counter}"
        claim_counter += 1
        
        # is_fraudulent Məntiqi Qaydası (Dələduzluq 5-7% olsun - biz 6% qoyuruq)
        is_fraudulent = np.random.binomial(1, 0.06)
        
        if is_fraudulent == 1:
            # Dələduzluq halıdırsa: Çox vaxt rədd edilir (Rejected) və həll vaxtı çox çəkir
            claim_status = np.random.choice(['Rejected', 'Pending'], p=[0.90, 0.10])
            settlement_time_days = np.random.randint(45, 91) # Vaxt uzundur (45-90 gün)
            claim_amount_requested = round(np.random.uniform(1500, 15000), 2)
        else:
            # Normal (təmiz) haldırsa: Təsdiqlənmə ehtimalı yüksəkdir, vaxt isə nisbətən qısadır
            claim_status = np.random.choice(['Approved', 'Rejected', 'Pending'], p=[0.70, 0.15, 0.15])
            # Vaxtı normal distribution ilə verib 1-90 arasına salırıq
            settlement_time_days = int(np.clip(np.random.normal(20, 15), 1, 90))
            claim_amount_requested = round(np.random.uniform(200, 5000), 2)
            
        # settled amount məntiqi: Requested-dən kiçik/bərabər olmalıdır. 
        # Rədd edilibsə və ya baxılmaqdadırsa 0 ödənilir.
        if claim_status == 'Approved':
            # Tələb olunan məbləğin 60-100%-i arası ödənilir
            claim_amount_settled = round(claim_amount_requested * np.random.uniform(0.6, 1.0), 2)
        else:
            claim_amount_settled = 0.0
            
        claims_data.append({
            'claim_id': claim_id,
            'policy_id': pid,
            'claim_amount_requested': claim_amount_requested,
            'claim_amount_settled': claim_amount_settled,
            'settlement_time_days': settlement_time_days,
            'claim_status': claim_status,
            'is_fraudulent': is_fraudulent
        })

claims_df = pd.DataFrame(claims_data)

# -------------------------------------------------------------
# 3. MƏLUMATLARIN CSV FAYLLARA YAZILMASI
# -------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent.parent
policies_csv_path = BASE_DIR / 'SIGORTA' / 'data' / 'policies_crm.csv'
claims_csv_path = BASE_DIR / 'SIGORTA' / 'data' / 'claims_data.csv'

policies_df.to_csv(policies_csv_path, index=False)
claims_df.to_csv(claims_csv_path, index=False)  

print(f"Data uğurla yaradıldı!")
print(f"Policies DataFrame sətir sayı: {len(policies_df)}")
print(f"Claims DataFrame sətir sayı: {len(claims_df)}")
print(f"Dələduzluq (Fraud) faizi: %{round(claims_df['is_fraudulent'].mean() * 100, 2)}")

print(f"Policies CSV yolu: {policies_csv_path}")
print(f"Claims CSV yolu: {claims_csv_path}")