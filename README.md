# 📊 KPI Forecast & Risk Tracker

[![Finance](https://img.shields.io/badge/Finance%20Analytics-KPI%20Tracker-blueviolet?style=for-the-badge)]()

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://kpi-risk-tracker-finance.streamlit.app/)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Prophet Forecasting](https://img.shields.io/badge/Forecasting-Prophet-0d1117?logo=meta&logoColor=white)](https://facebook.github.io/prophet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Author: Vikrant Thenge](https://img.shields.io/badge/Author-Vikrant%20Thenge-blueviolet)](https://www.linkedin.com/in/vthenge/)

---

A Streamlit-powered dashboard for forecasting financial KPIs and tracking portfolio risk using Prophet, yFinance, and Plotly.

---

## 🚀 Features

- 📈 Forecast stock prices using Prophet
- 📉 Visualize upper/lower risk bounds
- 📡 Pull live data from Yahoo Finance
- 📁 Upload your own CSV or use built-in demo
- 📊 Interactive Plotly charts
- 🔐 Modular layout for recruiter clarity

---

## 🧰 Tech Stack

| Tool        | Purpose                          |
|-------------|----------------------------------|
| Streamlit   | Frontend dashboard               |
| Prophet     | Time series forecasting          |
| yFinance    | Live stock data                  |
| Plotly      | Interactive visualizations       |
| Pandas      | Data wrangling                   |
| Pillow      | Logo/image handling              |

---

## 📦 Installation

```bash
git clone https://github.com/Vikrantthenge/financial-kpi-risk-tracker.git
cd financial-kpi-risk-tracker
pip install -r requirements.txt
streamlit run app.py
