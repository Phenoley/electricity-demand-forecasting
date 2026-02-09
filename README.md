# electricity-demand-forecasting
XGBoost &amp; LSTM forecasting for West African electricity

This project develops a machine learning–based forecasting and optimization framework for electricity demand and price dynamics in West African power systems. The study integrates macroeconomic indicators, climate variables, fuel prices, and power system capacity metrics to model real-world energy market behavior.

The work is designed to support:

Power system planning

Tariff analysis

Energy policy evaluation

Reliability and outage risk assessment

🎯 Objectives

Forecast electricity demand (MW) using machine learning models

Predict electricity prices (USD/kWh) under varying economic and supply conditions

Analyze the impact of capacity constraints, fuel prices, and climate variables

Provide a reproducible framework suitable for academic research and policy analysis

🧠 Models Implemented

XGBoost Regressor

Captures non-linear relationships between demand, price, and explanatory variables

LSTM Neural Network

Models temporal dependencies in electricity demand and pricing trends

📊 Dataset Description

The dataset represents synthetic but realistically calibrated electricity market data for selected West African countries.

Key Features

Macroeconomic indicators (GDP, inflation)

Climate variables (temperature, humidity, reservoir levels)

Power system capacity metrics

Fuel price dynamics

Market and reliability indicators

🧾 Data Dictionary
Feature	Description
country	West African country identifier
demand_mw	Electricity demand (MW)
price_usd_kwh	Electricity price (USD/kWh)
outage_risk_score	Probability-based index representing outage risk
gdp_usd_billion	Gross Domestic Product (Billion USD)
inflation_pct	Annual inflation rate (%)
electrification_pct	Percentage of population with electricity access
temp_celsius_mean	Mean ambient temperature (°C)
humidity_pct	Average humidity (%)
reservoir_level_pct	Hydropower reservoir level (%)
installed_capacity_mw	Total installed generation capacity (MW)
available_capacity_mw	Available generation capacity after outages (MW)
transmission_losses_pct	Percentage of power lost during transmission
gas_price_usd	Natural gas price (USD)
diesel_price_usd_liter	Diesel fuel price (USD/Liter)
fuel_price_index	Composite index of fuel costs
price_smoothing_factor	Regulatory price stabilization parameter
capacity_utilization	Ratio of demand to available capacity
reserve_margin_pct	Generation reserve margin (%)
