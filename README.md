# 🛢️ Oil Price Sentinel: Geopolitical Shock Forecasting

A Multi-Modal machine learning system designed to forecast Brent/WTI fluctuations by fusing GDELT geopolitical sentiment with market technicals.

## 🚀 Key Performance
- **Directional Accuracy:** ~55% (Standard)
- **Shock Accuracy (>3% moves):** 70.45%
- **Loss Function:** Pseudo-Huber (Shock-Optimized)

## 🛠️ Methodology
Based on "Novel algorithm to forecast crude oil price fluctuations incorporating news sentiment" (2019). We utilize:
1. **Sentiment Momentum:** Capturing the rate of change in global news.
2. **Feature Interpretation:** SHAP values to identify "Sentinel" country influence.
