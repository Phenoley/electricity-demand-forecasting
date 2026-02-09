import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

print("Generating West Africa Energy Dataset...")
print("=" * 60)

# Country configurations with realistic baselines
COUNTRIES = {
    'Nigeria': {'pop': 223, 'gdp': 477, 'elec': 62, 'installed': 12500, 'demand_base': 5500, 'price_base': 0.10, 'volatility': 1.4},
    'Ghana': {'pop': 34, 'gdp': 77, 'elec': 86, 'installed': 5100, 'demand_base': 3200, 'price_base': 0.175, 'volatility': 0.8},
    'Ivory Coast': {'pop': 29, 'gdp': 70, 'elec': 75, 'installed': 2300, 'demand_base': 1750, 'price_base': 0.14, 'volatility': 0.9},
    'Senegal': {'pop': 18, 'gdp': 28, 'elec': 70, 'installed': 1450, 'demand_base': 1000, 'price_base': 0.19, 'volatility': 0.85},
    'Mali': {'pop': 23, 'gdp': 19, 'elec': 50, 'installed': 900, 'demand_base': 500, 'price_base': 0.215, 'volatility': 1.1},
    'Burkina Faso': {'pop': 23, 'gdp': 19, 'elec': 22, 'installed': 350, 'demand_base': 300, 'price_base': 0.24, 'volatility': 1.15},
    'Guinea': {'pop': 14, 'gdp': 16, 'elec': 45, 'installed': 800, 'demand_base': 400, 'price_base': 0.20, 'volatility': 1.2},
    'Benin': {'pop': 13, 'gdp': 17, 'elec': 42, 'installed': 400, 'demand_base': 280, 'price_base': 0.18, 'volatility': 1.0},
    'Togo': {'pop': 9, 'gdp': 8, 'elec': 60, 'installed': 250, 'demand_base': 180, 'price_base': 0.22, 'volatility': 1.05},
    'Niger': {'pop': 26, 'gdp': 14, 'elec': 19, 'installed': 300, 'demand_base': 200, 'price_base': 0.25, 'volatility': 1.25},
    'Sierra Leone': {'pop': 8, 'gdp': 4, 'elec': 26, 'installed': 200, 'demand_base': 120, 'price_base': 0.28, 'volatility': 1.3},
    'Liberia': {'pop': 5, 'gdp': 4, 'elec': 28, 'installed': 150, 'demand_base': 90, 'price_base': 0.30, 'volatility': 1.35},
    'Mauritania': {'pop': 5, 'gdp': 9, 'elec': 50, 'installed': 200, 'demand_base': 130, 'price_base': 0.27, 'volatility': 1.1},
    'Guinea-Bissau': {'pop': 2, 'gdp': 2, 'elec': 35, 'installed': 50, 'demand_base': 35, 'price_base': 0.35, 'volatility': 1.4},
    'Cape Verde': {'pop': 0.6, 'gdp': 2, 'elec': 95, 'installed': 180, 'demand_base': 85, 'price_base': 0.32, 'volatility': 0.7}
}

# Generate hourly timestamps (2020-2024)
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31, 23, 0, 0)
hours = int((end_date - start_date).total_seconds() / 3600) + 1

data_list = []

for country, config in COUNTRIES.items():
    print(f"Processing {country}...")
    
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    n = len(timestamps)
    
    # Extract time features
    df = pd.DataFrame({'timestamp': timestamps})
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # GDP growth (2-6% annually)
    gdp_growth = 0.03 + np.random.rand() * 0.03
    years_elapsed = (df['year'] - 2020) + df['day_of_year'] / 365
    gdp = config['gdp'] * (1 + gdp_growth) ** years_elapsed
    
    # Electrification growth
    elec_growth = 0.015 + np.random.rand() * 0.015
    electrification = np.clip(config['elec'] + elec_growth * years_elapsed, 0, 100)
    
    # Installed capacity (gradual increases)
    capacity_growth = np.random.choice([0, 0, 0, 50, 100, 200], n)
    installed_capacity = config['installed'] + np.cumsum(capacity_growth)
    
    # Temperature patterns
    temp_base = 28 + np.random.randn() * 2
    temp_seasonal = 6 * np.sin(2 * np.pi * (df['day_of_year'] - 80) / 365)
    temp_daily = -4 * np.cos(2 * np.pi * df['hour'] / 24)
    temperature = temp_base + temp_seasonal + temp_daily + np.random.randn(n) * 1.5
    temperature = np.clip(temperature, 18, 42)
    
    # Humidity (inverse of temperature roughly)
    humidity = 75 - 1.2 * (temperature - 28) + 15 * np.sin(2 * np.pi * df['day_of_year'] / 365) + np.random.randn(n) * 8
    humidity = np.clip(humidity, 15, 95)
    
    # Reservoir levels (seasonal)
    reservoir_base = 70 + 20 * np.sin(2 * np.pi * (df['day_of_year'] - 150) / 365)
    reservoir = reservoir_base + np.random.randn(n) * 5
    reservoir = np.clip(reservoir, 25, 98)
    
    # Fuel prices
    gas_base = 4.5 + np.random.randn(n) * 0.5
    # Ukraine war spike (2022)
    ukraine_spike = np.where((df['year'] == 2022) & (df['month'] >= 3), 3.5, 0)
    gas_price = np.clip(gas_base + ukraine_spike + 0.3 * np.sin(2 * np.pi * df['day_of_year'] / 365), 2.5, 8.5)
    
    diesel_base = 1.2 + np.random.randn(n) * 0.1
    diesel_spike = np.where((df['year'] == 2022) & (df['month'] >= 3), 0.5, 0)
    diesel_price = np.clip(diesel_base + diesel_spike, 0.8, 1.8)
    
    fuel_index = 100 * (0.6 * gas_price / 4.5 + 0.4 * diesel_price / 1.2)
    
    # Inflation
    inflation_base = 8 + np.random.randn() * 2
    inflation_covid = np.where((df['year'] == 2020) & (df['month'] >= 3) & (df['month'] <= 8), 3, 0)
    inflation_ukraine = np.where((df['year'] == 2022) | (df['year'] == 2023), 4, 0)
    inflation = np.clip(inflation_base + inflation_covid + inflation_ukraine + np.random.randn(n) * 2, 5, 15)
    
    # Demand modeling
    demand_base = config['demand_base']
    
    # Temporal patterns
    hour_pattern = np.array([0.5, 0.48, 0.46, 0.45, 0.47, 0.65, 0.85, 0.92, 0.88, 0.82, 
                            0.80, 0.78, 0.80, 0.83, 0.85, 0.88, 0.93, 0.98, 1.0, 0.97, 
                            0.90, 0.82, 0.70, 0.58])
    demand_hourly = hour_pattern[df['hour']]
    
    # Weekly pattern
    weekly_pattern = np.array([1.0, 1.0, 0.98, 0.99, 1.0, 0.88, 0.75])
    demand_weekly = weekly_pattern[df['day_of_week']]
    
    # Seasonal pattern
    demand_seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (df['day_of_year'] - 80) / 365)
    
    # Growth trend
    demand_growth = 1.0 + 0.035 * years_elapsed
    
    # Temperature effect (non-linear, cooling demand)
    temp_effect = 1.0 + 0.4 * np.maximum(0, (temperature - 30) / 10)
    
    # GDP effect
    gdp_effect = (gdp / config['gdp']) ** 0.5
    
    # COVID impact
    covid_impact = np.ones(n)
    covid_mask = (df['year'] == 2020) & (df['month'] >= 3) & (df['month'] <= 8)
    covid_impact[covid_mask] = 0.7 - 0.2 * np.random.rand(covid_mask.sum())
    
    demand = (demand_base * demand_hourly * demand_weekly * demand_seasonal * 
              demand_growth * temp_effect * gdp_effect * covid_impact)
    
    # Add realistic noise (heteroscedastic)
    demand_noise = demand * config['volatility'] * 0.08 * np.random.randn(n)
    demand = np.maximum(50, demand + demand_noise)
    
    # Available capacity (maintenance, outages)
    availability_factor = 0.75 + 0.15 * np.random.rand(n)
    # Grid failures
    grid_failures = np.random.rand(n) < 0.001
    availability_factor[grid_failures] *= 0.6
    available_capacity = installed_capacity * availability_factor
    
    # Ensure demand doesn't exceed available capacity too much
    capacity_ratio = demand / available_capacity
    demand = np.where(capacity_ratio > 1.05, available_capacity * 0.98, demand)
    
    # Capacity utilization
    capacity_utilization = np.clip(demand / available_capacity, 0, 1)
    
    # Reserve margin
    reserve_margin = 100 * (available_capacity - demand) / demand
    
    # Transmission losses
    loss_base = 15 + config['volatility'] * 5
    transmission_losses = np.clip(loss_base + np.random.randn(n) * 3, 8, 28)
    
    # FIXED: Outage risk - realistic and no missing values
    # Base risk from capacity stress (exponential relationship)
    risk_capacity = capacity_utilization ** 1.5
    
    # Reserve margin effect (negative reserves = blackouts)
    risk_reserve = np.where(reserve_margin < 0, 0.9, 
                           np.where(reserve_margin < 5, 0.6,
                           np.where(reserve_margin < 10, 0.3, 0.1)))
    
    # Transmission quality
    risk_transmission = np.clip((transmission_losses - 10) / 40, 0, 0.3)
    
    # Infrastructure quality (country-specific baseline)
    infra_risk = config['volatility'] / 2
    
    # Time-based risk (higher at peak demand)
    peak_risk = np.where((df['hour'] >= 18) & (df['hour'] <= 21), 0.15, 0)
    
    # Weather stress (extreme heat increases risk)
    weather_risk = np.clip((temperature - 35) / 15, 0, 0.2)
    
    # Combine all factors
    outage_risk = np.clip(
        0.25 * risk_capacity + 0.35 * risk_reserve + 0.15 * risk_transmission + 
        0.15 * infra_risk + peak_risk + weather_risk + 
        0.05 * np.random.randn(n),
        0.01, 0.98
    )
    
    # Price modeling
    price_base = config['price_base']
    
    # Fuel price effect
    fuel_effect = (fuel_index / 100) ** 0.8
    
    # Scarcity pricing (capacity utilization effect)
    scarcity_premium = 1.0 + 1.2 * np.maximum(0, capacity_utilization - 0.75)
    
    # Reserve margin effect
    reserve_effect = 1.0 + np.clip(-0.015 * reserve_margin, -0.3, 0.5)
    
    # Transmission loss effect
    loss_effect = 1.0 + 0.02 * (transmission_losses - 15)
    
    # Inflation effect
    inflation_effect = (1 + inflation / 100) ** (years_elapsed)
    
    # Time of day premium
    peak_hours = (df['hour'] >= 18) & (df['hour'] <= 21)
    time_premium = np.where(peak_hours, 1.15, 1.0)
    
    # Subsidy reforms (random price jumps)
    subsidy_reform = np.ones(n)
    reform_events = np.random.rand(n) < 0.0005
    subsidy_reform[reform_events] = 1.5
    subsidy_reform = np.maximum.accumulate(subsidy_reform)
    
    price = (price_base * fuel_effect * scarcity_premium * reserve_effect * 
             loss_effect * inflation_effect * time_premium * subsidy_reform)
    
    # Add noise
    price_noise = price * 0.06 * np.random.randn(n)
    price = np.clip(price + price_noise, 0.05, 0.45)
    
    # Price smoothing factor
    price_smoothing = 0.85 + 0.15 * np.random.rand(n)
    
    # Compile country data
    country_data = pd.DataFrame({
        'timestamp': timestamps,
        'country': country,
        'demand_mw': demand,
        'price_usd_kwh': price,
        'outage_risk_score': outage_risk,
        'gdp_usd_billion': gdp,
        'inflation_pct': inflation,
        'electrification_pct': electrification,
        'temp_celsius_mean': temperature,
        'humidity_pct': humidity,
        'reservoir_level_pct': reservoir,
        'installed_capacity_mw': installed_capacity,
        'available_capacity_mw': available_capacity,
        'transmission_losses_pct': transmission_losses,
        'gas_price_usd': gas_price,
        'diesel_price_usd_liter': diesel_price,
        'fuel_price_index': fuel_index,
        'price_smoothing_factor': price_smoothing,
        'capacity_utilization': capacity_utilization,
        'reserve_margin_pct': reserve_margin
    })
    
    # FIXED: Introduce missing data - ONLY for sensor/weather data
    missing_rate = 0.05 + 0.03 * np.random.rand()
    missing_cols = ['temp_celsius_mean', 'humidity_pct', 'reservoir_level_pct']
    for col in missing_cols:
        mask = np.random.rand(n) < missing_rate
        country_data.loc[mask, col] = np.nan
    
    # Transmission losses - occasional missing (utility reporting delays)
    transmission_missing = np.random.rand(n) < 0.02
    country_data.loc[transmission_missing, 'transmission_losses_pct'] = np.nan
    
    data_list.append(country_data)

# Combine all countries
print("\nCombining data...")
final_df = pd.concat(data_list, ignore_index=True)

# Sort by timestamp and country
final_df = final_df.sort_values(['timestamp', 'country']).reset_index(drop=True)

# Round numerical columns
final_df['demand_mw'] = final_df['demand_mw'].round(2)
final_df['price_usd_kwh'] = final_df['price_usd_kwh'].round(4)
final_df['outage_risk_score'] = final_df['outage_risk_score'].round(3)
final_df['gdp_usd_billion'] = final_df['gdp_usd_billion'].round(2)
final_df['inflation_pct'] = final_df['inflation_pct'].round(2)
final_df['electrification_pct'] = final_df['electrification_pct'].round(2)
final_df['temp_celsius_mean'] = final_df['temp_celsius_mean'].round(2)
final_df['humidity_pct'] = final_df['humidity_pct'].round(2)
final_df['reservoir_level_pct'] = final_df['reservoir_level_pct'].round(2)
final_df['installed_capacity_mw'] = final_df['installed_capacity_mw'].round(2)
final_df['available_capacity_mw'] = final_df['available_capacity_mw'].round(2)
final_df['transmission_losses_pct'] = final_df['transmission_losses_pct'].round(2)
final_df['gas_price_usd'] = final_df['gas_price_usd'].round(3)
final_df['diesel_price_usd_liter'] = final_df['diesel_price_usd_liter'].round(3)
final_df['fuel_price_index'] = final_df['fuel_price_index'].round(2)
final_df['price_smoothing_factor'] = final_df['price_smoothing_factor'].round(3)
final_df['capacity_utilization'] = final_df['capacity_utilization'].round(3)
final_df['reserve_margin_pct'] = final_df['reserve_margin_pct'].round(2)

# Save to CSV
filename = 'west_africa_energy_data.csv'
final_df.to_csv(filename, index=False)

# Print statistics
print("\n" + "=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"\nFilename: {filename}")
print(f"Total rows: {len(final_df):,}")
print(f"Countries: {final_df['country'].nunique()}")
print(f"Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
print(f"File size: ~{len(final_df) * 250 / 1_000_000:.1f} MB")

print("\n" + "-" * 60)
print("VALUE RANGES:")
print("-" * 60)
print(f"Demand (MW): {final_df['demand_mw'].min():.0f} - {final_df['demand_mw'].max():.0f}")
print(f"Price ($/kWh): ${final_df['price_usd_kwh'].min():.3f} - ${final_df['price_usd_kwh'].max():.3f}")
print(f"Outage Risk: {final_df['outage_risk_score'].min():.3f} - {final_df['outage_risk_score'].max():.3f}")
print(f"Temperature (°C): {final_df['temp_celsius_mean'].min():.1f} - {final_df['temp_celsius_mean'].max():.1f}")
print(f"Reserve Margin (%): {final_df['reserve_margin_pct'].min():.1f} - {final_df['reserve_margin_pct'].max():.1f}")

print("\n" + "-" * 60)
print("SAMPLE CORRELATIONS:")
print("-" * 60)
corr_pairs = [
    ('capacity_utilization', 'outage_risk_score'),
    ('fuel_price_index', 'price_usd_kwh'),
    ('reserve_margin_pct', 'outage_risk_score'),
    ('temp_celsius_mean', 'demand_mw')
]
for col1, col2 in corr_pairs:
    corr = final_df[[col1, col2]].corr().iloc[0, 1]
    print(f"{col1} <-> {col2}: {corr:.3f}")

print("\n" + "-" * 60)
print("MISSING DATA:")
print("-" * 60)
missing_pct = (final_df.isnull().sum() / len(final_df) * 100)
missing_pct = missing_pct[missing_pct > 0]
if len(missing_pct) > 0:
    for col, pct in missing_pct.items():
        print(f"{col}: {pct:.2f}%")
else:
    print("No missing data in key operational metrics!")
    print("Weather sensors have realistic gaps.")

print("\n✓ Dataset ready for ML model training!")
print("  - XGBoost: Feature engineering complete")
print("  - LSTM: Temporal patterns embedded")
print("  - EDA: Rich correlation structure")
print("  - Outage Risk: Realistic, no missing values")
print("\nMetadata: Seed=42, Generated=" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))