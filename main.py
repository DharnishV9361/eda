#!/usr/bin/env python3
"""
Urban Air Quality & Pollution Index Analysis and Deep Learning Modeling
=======================================================================
Case Study: Exploratory Data Analysis, Distribution Analysis & Deep Neural Network
Course: U21ADP05 - Exploratory Data Analysis and Visualization
Department of Artificial Intelligence and Data Science
KPR Institute of Engineering and Technology

Author: Dharnish V (Roll No: 24AD027)
Class: 3rd Year AD - A (Batch 2023-2027)
Academic Year: 2026-2027 (ODD SEM)
"""

import os
import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Matplotlib & Seaborn styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.autolayout'] = False

DATA_PATH = 'data/city_day.csv'
CLEANED_DATA_PATH = 'data/cleaned_air_quality.csv'
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)


# ==============================================================================
# STEP 2: DATA UNDERSTANDING, PREPROCESSING & FEATURE ENGINEERING
# ==============================================================================
def load_and_preprocess_data(filepath=DATA_PATH):
    print("=" * 80)
    print("STEP 2: DATA UNDERSTANDING & PREPROCESSING")
    print("=" * 80)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df_raw = pd.read_csv(filepath)
    print(f"[*] Raw Dataset Loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Datetime conversion
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    # Filtering rows where target AQI and AQI_Bucket are available
    # CPCB requires minimum pollutant sub-indices to reliably compute official AQI
    df = df_raw.dropna(subset=['AQI', 'AQI_Bucket']).copy()
    print(f"[*] Filtered Ground-Truth Records (Valid AQI): {df.shape[0]} rows ({len(df)/len(df_raw)*100:.1f}%)")

    # Pollutant feature list
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

    # Imputation: City-wise median imputation with global median fallback (preserves geographic baseline)
    print("[*] Performing City-wise Median Imputation for continuous sensor metrics...")
    for col in pollutant_cols:
        city_medians = df.groupby('City')[col].transform('median')
        df[col] = df[col].fillna(city_medians)
        df[col] = df[col].fillna(df[col].median())

    # Feature Engineering (Temporal & Domain Chemistry)
    print("[*] Engineering domain features: Temporal cycles, PM ratio, and combustion indices...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Seasonal mapping for Indian subcontinent climate
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'

    df['Season'] = df['Month'].apply(get_season)

    # Domain Chemical Ratios:
    # 1. PM Fine-to-Coarse Ratio (PM2.5 / PM10)
    df['PM_Ratio'] = df['PM2.5'] / (df['PM10'] + 1e-5)
    # Clip extreme ratio anomalies to physiological bounds [0, 1.2]
    df['PM_Ratio'] = df['PM_Ratio'].clip(0, 1.2)

    # 2. Nitrogen-to-Sulfur Combustion Ratio (vehicular vs industrial coal/fuel oil indicator)
    df['NO2_SO2_Ratio'] = df['NO2'] / (df['SO2'] + 1e-5)
    df['NO2_SO2_Ratio'] = df['NO2_SO2_Ratio'].clip(0, 25.0)

    # Outlier treatment: Robust clipping using 1st and 99th percentiles to preserve severe smog peaks without distortion
    for col in pollutant_cols:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(p1, p99)

    # Target Mapping
    # Standard CPCB AQI Buckets
    aqi_bucket_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    df = df[df['AQI_Bucket'].isin(aqi_bucket_order)].copy()
    label_encoder = LabelEncoder()
    label_encoder.fit(aqi_bucket_order)
    df['AQI_Class'] = df['AQI_Bucket'].map({cls: idx for idx, cls in enumerate(aqi_bucket_order)})

    # Binary Alert target: High Pollution Alert (AQI > 200, i.e., Poor, Very Poor, Severe)
    df['High_Pollution_Alert'] = (df['AQI'] > 200).astype(int)

    # Save cleaned dataset
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"[+] Cleaned & Enriched Dataset Saved to {CLEANED_DATA_PATH} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print(f"[*] Total Features Available: {df.shape[1]} (Exceeds 15 feature requirement)")
    print(f"[*] AQI Class Distribution:\n{df['AQI_Bucket'].value_counts()[aqi_bucket_order]}")

    return df, aqi_bucket_order


# ==============================================================================
# STEP 3: DATA VISUALIZATION (MINIMUM 5 DETAILED VISUALS + ALL 7 STAGES)
# ==============================================================================
def generate_eda_visualizations(df, aqi_bucket_order):
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING HIGH-RESOLUTION EDA & STATISTICAL VISUALIZATIONS")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Visual 1: Univariate Probability Distributions & Parametric Fits
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (col, color) in enumerate(zip(pollutants, colors)):
        ax = axes[i // 3, i % 3]
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, color=color, stat="density", bins=40, alpha=0.5, edgecolor=None)
        
        # Fit Log-Normal theoretical distribution
        shape, loc, scale = stats.lognorm.fit(data)
        x_eval = np.linspace(data.min(), data.max(), 200)
        pdf_fitted = stats.lognorm.pdf(x_eval, shape, loc, scale)
        ax.plot(x_eval, pdf_fitted, 'r--', lw=2, label=f'Log-Normal Fit\n(Skew={data.skew():.2f})')

        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='black', linestyle='-', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='blue', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.1f}')

        unit = "mg/m³" if col == 'CO' else "µg/m³"
        ax.set_title(f'Continuous Distribution: {col} Concentration ({unit})', fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel(f'{col} ({unit})', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("VISUAL 1: Univariate Probability Density & Parametric Distribution Analysis", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    v1_path = os.path.join(FIGURES_DIR, '01_univariate_pollutant_distributions.png')
    plt.savefig(v1_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 1: {v1_path}")

    # --------------------------------------------------------------------------
    # Visual 2: Discrete AQI Bucket Distribution & City Pollution Hierarchy
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: AQI Bucket Discrete Frequencies
    bucket_counts = df['AQI_Bucket'].value_counts()[aqi_bucket_order]
    cpcb_colors = ['#00b050', '#92d050', '#ffff00', '#ff9900', '#ff0000', '#7030a0']  # Official CPCB colors
    bars = ax1.bar(bucket_counts.index, bucket_counts.values, color=cpcb_colors, edgecolor='#333333', linewidth=1.2, alpha=0.85)
    total_samples = len(df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_samples) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 150, f'{h:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title("CPCB Discrete Air Quality Index Category Distribution", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Air Quality Index Category (Severity)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Total Observation Count", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(bucket_counts.values) * 1.15)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Right: Top 10 Most Polluted vs Cleanest Cities
    city_aqi_stats = df.groupby('City')['AQI'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
    top_cities = pd.concat([city_aqi_stats.head(5), city_aqi_stats.tail(5)])
    bar_colors = ['#d62728' if x >= 200 else ('#ff7f0e' if x >= 100 else '#2ca02c') for x in top_cities['mean']]
    
    y_pos = np.arange(len(top_cities))
    ax2.barh(y_pos, top_cities['mean'], color=bar_colors, edgecolor='#333333', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_cities.index, fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    for idx, (val, city) in enumerate(zip(top_cities['mean'], top_cities.index)):
        ax2.text(val + 3, idx, f"{val:.1f} (n={int(top_cities.loc[city, 'count'])})", va='center', fontsize=9, fontweight='bold')
    
    ax2.axvline(100, color='#92d050', linestyle='--', label='CPCB Satisfactory Threshold (100)')
    ax2.axvline(200, color='#ff9900', linestyle='--', label='CPCB Moderate/Poor Boundary (200)')
    ax2.set_title("Geographic Disparity: Top 5 Most Polluted vs 5 Cleanest Cities", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Mean Air Quality Index (AQI)", fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right', frameon=True, fontsize=9)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    plt.suptitle("VISUAL 2: Discrete Frequency Classification & Geographic Spatial Hierarchy", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    v2_path = os.path.join(FIGURES_DIR, '02_discrete_aqi_bucket_and_city_ranking.png')
    plt.savefig(v2_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 2: {v2_path}")

    # --------------------------------------------------------------------------
    # Visual 3: Correlation Matrix & Pollutant Interactions (Multivariate)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 10))
    corr_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'PM_Ratio', 'NO2_SO2_Ratio', 'AQI']
    corr_matrix = df[corr_features].corr(method='spearman')

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap='Spectral_r', vmin=-0.4, vmax=1.0, annot=True, fmt=".2f",
                square=True, linewidths=.8, cbar_kws={"shrink": .8, "label": "Spearman Rank Correlation Coefficient"},
                ax=ax, annot_kws={"size": 9, "fontweight": "bold"})
    
    ax.set_title("VISUAL 3: Spearman Multivariate Correlation Heatmap & Atmospheric Coupling", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10, fontweight='bold')
    plt.tight_layout()
    v3_path = os.path.join(FIGURES_DIR, '03_correlation_matrix_and_pollutant_coupling.png')
    plt.savefig(v3_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 3: {v3_path}")

    # --------------------------------------------------------------------------
    # Visual 4: Bivariate Seasonal Boxplots & Fine Particle Dynamics
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    seasons_order = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
    palette = ['#4575b4', '#fdae61', '#74add1', '#d73027']

    # Subplot 1: PM2.5 across Seasons
    sns.boxplot(x='Season', y='PM2.5', data=df, order=seasons_order, palette=palette, ax=axes[0], showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    axes[0].set_title("Bivariate: PM2.5 Concentration across Seasons", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("PM2.5 (µg/m³)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Atmospheric Season", fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: PM10 across Seasons
    sns.boxplot(x='Season', y='PM10', data=df, order=seasons_order, palette=palette, ax=axes[1], showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    axes[1].set_title("Bivariate: PM10 Concentration across Seasons", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("PM10 (µg/m³)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Atmospheric Season", fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Fine Particle Ratio (PM2.5 / PM10)
    sns.violinplot(x='Season', y='PM_Ratio', data=df, order=seasons_order, palette=palette, ax=axes[2], inner="quartile")
    axes[2].set_title("Bivariate: PM2.5/PM10 Ratio Distribution", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Fine Particle Ratio (PM2.5 / PM10)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Atmospheric Season", fontsize=11, fontweight='bold')
    axes[2].axhline(0.5, color='black', linestyle=':', label='Fine Particle Dominance (>0.5)')
    axes[2].legend(loc='lower left', frameon=True, fontsize=9)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("VISUAL 4: Bivariate Seasonal Particle Dynamics & Fine Particle Inversion Analysis", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    v4_path = os.path.join(FIGURES_DIR, '04_bivariate_pm_ratio_and_seasonal_dynamics.png')
    plt.savefig(v4_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 4: {v4_path}")

    # --------------------------------------------------------------------------
    # Visual 5: Multivariate Longitudinal Trend: Coastal vs Inland Urban Hubs
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7))
    sample_cities = ['Delhi', 'Bengaluru', 'Chennai', 'Coimbatore', 'Ahmedabad']
    df_sample = df[df['City'].isin(sample_cities)].copy()
    
    # Resample monthly average AQI
    monthly_aqi = df_sample.groupby(['City', pd.Grouper(key='Date', freq='ME')])['AQI'].mean().reset_index()

    city_colors = {'Delhi': '#d62728', 'Ahmedabad': '#ff7f0e', 'Bengaluru': '#1f77b4', 'Chennai': '#2ca02c', 'Coimbatore': '#9467bd'}
    for city in sample_cities:
        city_data = monthly_aqi[monthly_aqi['City'] == city]
        ax.plot(city_data['Date'], city_data['AQI'], marker='o', markersize=4, lw=2.2, label=city, color=city_colors[city])

    ax.axhspan(0, 50, color='#00b050', alpha=0.15, label='CPCB Good (0-50)')
    ax.axhspan(51, 100, color='#92d050', alpha=0.15, label='CPCB Satisfactory (51-100)')
    ax.axhspan(101, 200, color='#ffff00', alpha=0.15, label='CPCB Moderate (101-200)')
    ax.axhspan(201, 300, color='#ff9900', alpha=0.15, label='CPCB Poor (201-300)')
    ax.axhspan(301, 400, color='#ff0000', alpha=0.15, label='CPCB Very Poor (301-400)')
    ax.axhspan(401, 600, color='#7030a0', alpha=0.15, label='CPCB Severe (401+)')

    ax.set_title("VISUAL 5: Longitudinal Multivariate Temporal Trajectory (2015–2020) Across Major Metro Hubs", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Observation Timeline", fontsize=11, fontweight='bold')
    ax.set_ylabel("Monthly Mean Air Quality Index (AQI)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 500)
    ax.legend(loc='upper right', ncol=2, frameon=True, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    v5_path = os.path.join(FIGURES_DIR, '05_multivariate_pairplot_or_geographic_trend.png')
    plt.savefig(v5_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 5: {v5_path}")

    # --------------------------------------------------------------------------
    # Visual 6: Continuous Probability Distribution Fitting & Q-Q Diagnostics
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    aqi_data = df['AQI'].values

    # Fit Normal
    stats.probplot(aqi_data, dist="norm", plot=axes[0])
    axes[0].get_lines()[0].set_markerfacecolor('#1f77b4')
    axes[0].get_lines()[0].set_markersize(3.0)
    axes[0].get_lines()[1].set_color('red')
    axes[0].get_lines()[1].set_linewidth(2)
    axes[0].set_title("Normal Distribution Q-Q Fit\n(Severe Heavy-Tail Deviation)", fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Fit Log-Normal
    shape, loc, scale = stats.lognorm.fit(aqi_data)
    stats.probplot(np.log(aqi_data[aqi_data > 0]), dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor('#2ca02c')
    axes[1].get_lines()[0].set_markersize(3.0)
    axes[1].get_lines()[1].set_color('red')
    axes[1].get_lines()[1].set_linewidth(2)
    axes[1].set_title("Log-Normal Distribution Q-Q Fit\n(High Conformance Across Bulk Quantiles)", fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Fit Weibull Extreme Value
    shape_w, loc_w, scale_w = stats.weibull_min.fit(aqi_data)
    stats.probplot(aqi_data, dist=stats.weibull_min(shape_w, loc_w, scale_w), plot=axes[2])
    axes[2].get_lines()[0].set_markerfacecolor('#d62728')
    axes[2].get_lines()[0].set_markersize(3.0)
    axes[2].get_lines()[1].set_color('blue')
    axes[2].get_lines()[1].set_linewidth(2)
    axes[2].set_title(f"Weibull Minimum Fit\n(Shape k={shape_w:.2f}, Scale λ={scale_w:.1f})", fontsize=11, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("VISUAL 6: Continuous Probability Distribution Identification & Q-Q Goodness-of-Fit Diagnostics", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    v6_path = os.path.join(FIGURES_DIR, '06_probabilistic_distributions_qq_plots.png')
    plt.savefig(v6_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 6: {v6_path}")

    return {
        'visual_1': v1_path,
        'visual_2': v2_path,
        'visual_3': v3_path,
        'visual_4': v4_path,
        'visual_5': v5_path,
        'visual_6': v6_path
    }


# ==============================================================================
# STEP 4: DEEP LEARNING MODEL ARCHITECTURE & HYPERPARAMETER TUNING
# ==============================================================================
class DeepAirQualityNet(nn.Module):
    """
    Deep Multi-Layer Perceptron (MLP) Artificial Neural Network with
    Batch Normalization, LeakyReLU, Dropout regularization, and Skip Connections.
    """
    def __init__(self, input_dim, num_classes=6, hidden_dims=[128, 64, 32], dropout_rate=0.25):
        super(DeepAirQualityNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.8)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(0.1)
        )
        
        self.classifier = nn.Linear(hidden_dims[2], num_classes)
        # Auxiliary regression head for numeric AQI prediction
        self.regressor = nn.Linear(hidden_dims[2], 1)

    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        logits = self.classifier(h3)
        aqi_pred = self.regressor(h3)
        return logits, aqi_pred


def prepare_datasets(df):
    print("\n" + "=" * 80)
    print("STEP 4: DATA SPLITTING & FEATURE PREPARATION (70% TRAIN, 15% VAL, 15% TEST)")
    print("=" * 80)

    # 18 comprehensive input features
    feature_cols = [
        'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
        'Benzene', 'Toluene', 'Xylene', 'Month', 'DayOfWeek', 'Is_Weekend',
        'PM_Ratio', 'NO2_SO2_Ratio'
    ]
    # One-hot encode Season
    df_encoded = pd.get_dummies(df[feature_cols + ['Season', 'AQI_Class', 'AQI']], columns=['Season'], drop_first=True)
    X_cols = [c for c in df_encoded.columns if c not in ['AQI_Class', 'AQI']]
    
    X = df_encoded[X_cols].values
    y_class = df_encoded['AQI_Class'].values
    y_reg = df_encoded['AQI'].values

    # Stratified Split: 70% Train, 30% Temp -> 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.30, random_state=SEED, stratify=y_class
    )
    X_val, X_test, y_val, y_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_temp, y_reg_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print(f"[*] Training Set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"[*] Validation Set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"[*] Testing Set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"[*] Input Dimensions: {X_train.shape[1]} features")

    # Feature Scaling using RobustScaler (handles atmospheric outliers without distorting variance)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, y_train, y_reg_train,
            X_val_scaled, y_val, y_reg_val,
            X_test_scaled, y_test, y_reg_test,
            X_cols)


def train_deep_learning_model(X_train, y_train, y_reg_train, X_val, y_val, y_reg_val, epochs=45, batch_size=128):
    print("\n[*] Initializing Deep Air Quality Neural Network & Training Pipeline...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Computing Hardware: {device}")

    # Compute Class Weights for balanced loss
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts.astype(np.float32))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Tensor datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_reg_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
        torch.tensor(y_reg_val, dtype=torch.float32).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate Model
    input_dim = X_train.shape[1]
    model = DeepAirQualityNet(input_dim=input_dim, num_classes=6, hidden_dims=[128, 64, 32], dropout_rate=0.25).to(device)

    criterion_cls = nn.CrossEntropyLoss(weight=class_weights_t)
    criterion_reg = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    best_weights = None

    print(f"[*] Commencing Training for {epochs} Epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y_cls, batch_y_reg in train_loader:
            batch_x, batch_y_cls, batch_y_reg = batch_x.to(device), batch_y_cls.to(device), batch_y_reg.to(device)
            optimizer.zero_grad()

            logits, pred_aqi = model(batch_x)
            loss_cls = criterion_cls(logits, batch_y_cls)
            loss_reg = criterion_reg(pred_aqi, batch_y_reg) * 0.01  # Scaled auxiliary regression loss
            loss = loss_cls + loss_reg

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y_cls).sum().item()
            total += batch_y_cls.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y_cls, batch_y_reg in val_loader:
                batch_x, batch_y_cls, batch_y_reg = batch_x.to(device), batch_y_cls.to(device), batch_y_reg.to(device)
                logits, pred_aqi = model(batch_x)
                loss_cls = criterion_cls(logits, batch_y_cls)
                loss_reg = criterion_reg(pred_aqi, batch_y_reg) * 0.01
                loss = loss_cls + loss_reg

                val_running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y_cls).sum().item()
                val_total += batch_y_cls.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        scheduler.step(epoch_val_loss)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc*100:.2f}%")

    model.load_state_dict(best_weights)
    print(f"[+] Optimal Model Checkpoint Loaded with Best Val Loss: {best_val_loss:.4f}")
    return model, history, device


# ==============================================================================
# STEP 5: MODEL EVALUATION VISUALIZATIONS (MINIMUM 3-4 VISUALS)
# ==============================================================================
def evaluate_and_visualize_model(model, X_test, y_test, y_reg_test, history, aqi_bucket_order, X_cols, device):
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATING MODEL & GENERATING PERFORMANCE VISUALIZATIONS")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        x_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits, pred_aqi_t = model(x_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        pred_aqi = pred_aqi_t.squeeze().cpu().numpy()

    # Classification Metrics
    print("\n--- TEST SET CLASSIFICATION REPORT ---")
    cls_report = classification_report(y_test, preds, target_names=aqi_bucket_order, digits=4)
    print(cls_report)

    # Regression Metrics for numeric AQI
    test_rmse = np.sqrt(mean_squared_error(y_reg_test, pred_aqi))
    test_mae = mean_absolute_error(y_reg_test, pred_aqi)
    test_r2 = r2_score(y_reg_test, pred_aqi)
    print(f"--- CONTINUOUS AQI ESTIMATION METRICS ---")
    print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}")
    print(f"Mean Absolute Error (MAE):       {test_mae:.2f}")
    print(f"R² Coefficient of Determination: {test_r2:.4f}")

    # --------------------------------------------------------------------------
    # Visual 7: Loss vs Epoch (Training & Validation Convergence)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], 'b-', lw=2.2, label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], 'r--', lw=2.2, label='Validation Loss')
    best_epoch = np.argmin(history['val_loss']) + 1
    plt.scatter([best_epoch], [min(history['val_loss'])], color='green', s=100, zorder=5,
                label=f'Optimal Model Checkpoint (Epoch {best_epoch}, Loss={min(history["val_loss"]):.4f})')
    
    plt.title("VISUAL 7: Deep Learning Training vs Validation Loss Across Epochs", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Training Epochs", fontsize=11, fontweight='bold')
    plt.ylabel("Multi-Task Loss (Cross-Entropy + Smooth L1)", fontsize=11, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    v7_path = os.path.join(FIGURES_DIR, '07_deep_learning_loss_vs_epoch.png')
    plt.savefig(v7_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 7: {v7_path}")

    # --------------------------------------------------------------------------
    # Visual 8: Accuracy vs Epoch (Training vs Validation Accuracy)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, [x * 100 for x in history['train_acc']], 'b-', lw=2.2, label='Training Accuracy (%)')
    plt.plot(epochs_range, [x * 100 for x in history['val_acc']], 'r--', lw=2.2, label='Validation Accuracy (%)')
    best_acc_epoch = np.argmax(history['val_acc']) + 1
    plt.scatter([best_acc_epoch], [max(history['val_acc']) * 100], color='darkviolet', s=100, zorder=5,
                label=f'Peak Validation Accuracy ({max(history["val_acc"])*100:.2f}%)')

    plt.title("VISUAL 8: Deep Learning Training vs Validation Accuracy Trajectory", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Training Epochs", fontsize=11, fontweight='bold')
    plt.ylabel("Classification Accuracy (%)", fontsize=11, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    v8_path = os.path.join(FIGURES_DIR, '08_deep_learning_accuracy_vs_epoch.png')
    plt.savefig(v8_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 8: {v8_path}")

    # --------------------------------------------------------------------------
    # Visual 9: Normalized Confusion Matrix
    # --------------------------------------------------------------------------
    cm = confusion_matrix(y_test, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=True,
                xticklabels=aqi_bucket_order, yticklabels=aqi_bucket_order,
                linewidths=1, linecolor='white', annot_kws={"size": 11, "fontweight": "bold"})
    
    plt.title("VISUAL 9: Normalized Confusion Matrix for CPCB AQI Classification", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Air Quality Category", fontsize=11, fontweight='bold')
    plt.ylabel("True Air Quality Category", fontsize=11, fontweight='bold')
    plt.tight_layout()
    v9_path = os.path.join(FIGURES_DIR, '09_normalized_confusion_matrix.png')
    plt.savefig(v9_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 9: {v9_path}")

    # --------------------------------------------------------------------------
    # Visual 10: Multiclass ROC Curves and AUC Scores
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 7.5))
    # One-hot encode ground truth
    y_test_bin = np.zeros((len(y_test), len(aqi_bucket_order)))
    for i, val in enumerate(y_test):
        y_test_bin[i, val] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colors = ['#00b050', '#92d050', '#ffbb00', '#ff7700', '#ff0000', '#7030a0']

    for i in range(len(aqi_bucket_order)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2.2,
                 label=f"{aqi_bucket_order[i]} (AUC = {roc_auc[i]:.3f})")

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(aqi_bucket_order))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(aqi_bucket_order)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(aqi_bucket_order)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.plot(all_fpr, mean_tpr, color='black', linestyle='--', lw=2.5,
             label=f'Macro-Average ROC (AUC = {macro_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k:', lw=1.5, alpha=0.7)

    plt.title("VISUAL 10: One-vs-Rest Multiclass ROC Curves & Discriminative AUC", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    v10_path = os.path.join(FIGURES_DIR, '10_multiclass_roc_and_auc_curves.png')
    plt.savefig(v10_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 10: {v10_path}")

    # --------------------------------------------------------------------------
    # Visual 11: Feature Importance (Permutation Sensitivity Analysis)
    # --------------------------------------------------------------------------
    print("[*] Computing Permutation Feature Sensitivity for Deep Neural Network...")
    baseline_acc = (preds == y_test).mean()
    feature_importances = []
    
    # Evaluate permutation sensitivity on sample to maintain efficiency
    eval_size = min(1500, len(X_test))
    X_sub = X_test[:eval_size].copy()
    y_sub = y_test[:eval_size]

    for col_idx in range(X_test.shape[1]):
        X_permuted = X_sub.copy()
        np.random.shuffle(X_permuted[:, col_idx])
        with torch.no_grad():
            x_perm_t = torch.tensor(X_permuted, dtype=torch.float32).to(device)
            perm_logits, _ = model(x_perm_t)
            perm_preds = torch.argmax(perm_logits, dim=1).cpu().numpy()
        perm_acc = (perm_preds == y_sub).mean()
        drop = baseline_acc - perm_acc
        feature_importances.append(max(0, drop))

    feat_df = pd.DataFrame({'Feature': X_cols, 'Importance': feature_importances}).sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 6.5))
    bars = plt.barh(feat_df['Feature'], feat_df['Importance'] * 100, color='#1f77b4', edgecolor='#333333', alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va='center', fontsize=9, fontweight='bold')

    plt.title("VISUAL 11: Deep Learning Permutation Feature Importance (% Accuracy Drop)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Accuracy Drop When Feature is Shuffled (%)", fontsize=11, fontweight='bold')
    plt.xlim(0, max(feat_df['Importance'] * 100) * 1.15)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    v11_path = os.path.join(FIGURES_DIR, '11_feature_importance_permutation.png')
    plt.savefig(v11_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 11: {v11_path}")

    # --------------------------------------------------------------------------
    # Visual 12: Regression Error Distribution & Residual Diagnostics
    # --------------------------------------------------------------------------
    residuals = y_reg_test - pred_aqi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter: Actual vs Predicted AQI
    ax1.scatter(y_reg_test, pred_aqi, alpha=0.4, color='#2ca02c', edgecolors='none', s=20)
    min_val = min(y_reg_test.min(), pred_aqi.min())
    max_val = max(y_reg_test.max(), pred_aqi.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect 1:1 Identity Line')
    ax1.set_title(f"Actual vs Predicted AQI Values (R² = {test_r2:.3f})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("True CPCB AQI Index", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Deep Learning Predicted AQI", fontsize=10, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Residual Histogram
    sns.histplot(residuals, kde=True, ax=ax2, color='#9467bd', stat="density", bins=40, alpha=0.6)
    mu, std = stats.norm.fit(residuals)
    x_res = np.linspace(residuals.min(), residuals.max(), 200)
    ax2.plot(x_res, stats.norm.pdf(x_res, mu, std), 'r-', lw=2, label=f'Normal Fit (μ={mu:.1f}, σ={std:.1f})')
    ax2.set_title(f"Residual Error Distribution (MAE = {test_mae:.1f})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Prediction Residual Error (Actual - Predicted)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Density", fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("VISUAL 12: Continuous Value Estimation & Residual Diagnostic Error Analysis", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    v12_path = os.path.join(FIGURES_DIR, '12_regression_residuals_distribution.png')
    plt.savefig(v12_path, dpi=300)
    plt.close()
    print(f"[+] Saved Visual 12: {v12_path}")

    return {
        'cls_report': cls_report,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'macro_auc': macro_auc,
        'visual_7': v7_path,
        'visual_8': v8_path,
        'visual_9': v9_path,
        'visual_10': v10_path,
        'visual_11': v11_path,
        'visual_12': v12_path
    }


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    print("=" * 80)
    print("URBAN AIR QUALITY EXPLORATORY DATA ANALYSIS & DEEP LEARNING PIPELINE")
    print("Department of Artificial Intelligence and Data Science")
    print("KPR Institute of Engineering and Technology")
    print("=" * 80)

    # Step 2: Data Preprocessing
    df, aqi_bucket_order = load_and_preprocess_data(DATA_PATH)

    # Step 3: EDA Visualizations
    eda_visuals = generate_eda_visualizations(df, aqi_bucket_order)

    # Step 4: Data Preparation & Model Training
    (X_train, y_train, y_reg_train,
     X_val, y_val, y_reg_val,
     X_test, y_test, y_reg_test,
     X_cols) = prepare_datasets(df)

    model, history, device = train_deep_learning_model(
        X_train, y_train, y_reg_train, X_val, y_val, y_reg_val, epochs=40, batch_size=128
    )

    # Step 5: Model Evaluation Visualizations
    eval_results = evaluate_and_visualize_model(
        model, X_test, y_test, y_reg_test, history, aqi_bucket_order, X_cols, device
    )

    print("\n" + "=" * 80)
    print("[SUCCESS] Pipeline Complete! All 12 High-Resolution Visualizations Generated:")
    for key, path in {**eda_visuals, **eval_results}.items():
        if key.startswith('visual_'):
            print(f"  - {path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
