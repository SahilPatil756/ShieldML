# ShieldML Data Management & Provenance

## Expected Dataset Format
ShieldML models financial transactions. The standard format follows the canonical Credit Card Fraud benchmark:
- **`Time`**: Seconds elapsed between this transaction and the first transaction in the dataset (numeric, 0 to 172,800).
- **`V1` through `V28`**: Principal components obtained with PCA (numeric, continuous).
- **`Amount`**: Transaction amount in USD (numeric, continuous $\ge 0.0$).
- **`Class`**: Ground truth target label (`0` = Legitimate, `1` = Fraudulent).

## Automated Benchmark Dataset Generation
If the raw dataset is not present at `data/raw/creditcard.csv`, ShieldML automatically invokes `generate_creditcard_dataset` from `shieldml.data.synthetic`. This generates a high-fidelity 15,000-row dataset preserving:
- Heavy class imbalance ($124:1$ ratio, $\sim 0.8\%$ fraud rate).
- Diurnal time frequencies with night-time fraud peaks.
- Log-normal transaction amounts with micro-charge card testing ($<\$5$) and large cashouts ($>\$1000$).
- Characteristic PCA shifts in fraud-discriminating components ($V14, V12, V4, V10, V17$).

## Downloading the Public Kaggle Dataset
To train on the 284,807-row Kaggle dataset:
```bash
# Using Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/
unzip data/raw/creditcardfraud.zip -d data/raw/
```
The ingestion pipeline automatically detects and processes the full dataset without any code changes.
