# MNQ Machine Learning Quantitative Trading Strategy

A walk-forward optimized (WFO) machine learning trading system built for **Micro E-mini Nasdaq Futures (MNQ)**, utilizing LightGBM optimized directly for **Profit Factor gain**.

## Strategy Overview
This project compares two model variants to evaluate temporal regime effects on predictive alpha:
1. **Hour-Inclusive Model:** Incorporates session hour metadata to capture time-of-day volatility regimes.
2. **No-Hour Model:** Relies purely on structural volume shocks and technical indicators without temporal bias.

## Key Features & Artifacts
* **Objective Function:** Custom Profit Factor optimization rather than raw log-loss classification.
* **Feature Engineering:** Technical indicators (RSI, MACD, ATR, EMAs, SMAs) combined with multi-period relative volume lags.
* **Visual Diagnostics:** Automated descending-order feature importance gain comparisons stored in `assets/`.

## Repository Structure