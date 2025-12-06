# =============================================
# 📘 Step 1: Install SDV (CTGAN-based library)
# =============================================
# !pip install sdv pandas scikit-learn

# =============================================
# 📘 Step 2: Import dependencies
# =============================================
import pandas as pd
from sdv.tabular import CTGAN

# =============================================
# 📘 Step 3: Load your real dataset
# =============================================
# Replace with your real CSV path
real_df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

print("✅ Original dataset shape:", real_df.shape)
print(real_df.head())

# =============================================
# 📘 Step 4: Train CTGAN model on your data
# =============================================
# Initialize CTGAN model
model = CTGAN(
    epochs=300,          # Train longer for better quality
    batch_size=32,
    verbose=True
)

# Fit on real data
model.fit(real_df)

# =============================================
# 📘 Step 5: Generate synthetic data
# =============================================
# You can choose how many rows you want
synthetic_data = model.sample(10000)   # e.g., generate 10,000 new samples

print("✅ Synthetic dataset shape:", synthetic_data.shape)
print(synthetic_data.head())

# =============================================
# 📘 Step 6: Save synthetic dataset
# =============================================
synthetic_data.to_csv("synthetic_disease_dataset.csv", index=False)
print("💾 Saved as 'synthetic_disease_dataset.csv'")
