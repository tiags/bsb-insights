# bsb-insights

# BSB Insights — Daily Market Analysis Pipeline

**Balance Sheet Buddy (BSB)** is a modular, end-to-end system that:
- Scrapes financial, sentiment, and options data for 700+ equities
- Feeds an ML model (Random Forest / XGBoost) for trade signal generation
- Outputs visual insights to a live Tableau dashboard

**View the Dashboard:** 
[Daily Market Insights on Tableau](https://public.tableau.com/app/profile/tiago.abreu/viz/DailyMarketInsightsPriceActionMomentum/EquityInsightsDashboard?publish=yes)


---

##  Overview

This repository contains two key components:

- `bsb_scraper.py`: Collects and processes daily financial and technical data from sources like `yfinance`, `stockdex`, `finnhub`, and RSS feeds. Saves results to SQLite and CSV for downstream analysis.
- `ml_model.py`: Loads the scraped dataset and applies machine learning to identify **bullish and bearish trade setups** using engineered features.
---

##  Features

- Over 50 features scraped daily
- Modular functions for easy testing and extension
- > 100 features and calculations used for modelling
- Outputs predictions + entry/exit strategy for Tableau ingestion
- Dual-target modeling (bullish + bearish)
- Grid-searched RF & XGBoost classifiers with calibrated probabilities
- Model evaluation with precision-tuned thresholds
- Export-ready DataFrame with entry/exit recommendations

---

## Files

- `ml_model.py` — Primary script, modular ML pipeline
- `bsb_scraper.py` — Data ingestion and feature extraction pipeline
- `README.md` — This file

---

## Note

This repo is a **partial redacted version** of a larger private trading pipeline. It is shared for demonstration and portfolio purposes.

---
