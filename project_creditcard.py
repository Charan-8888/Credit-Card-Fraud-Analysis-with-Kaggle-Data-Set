import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

df = pd.read_csv(
    "creditcard.csv"
)

###########################
#### M1 Data Cleansing ####
###########################

# Drop completely empty columns and rows
df.dropna(how="all", axis=1, inplace=True)
df.dropna(how="all", axis=0, inplace=True)

# Remove any 'Unnamed' columns if they appear
df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)],
        errors="ignore", inplace=True)

# Check and remove duplicate rows
n_before = len(df)
df.drop_duplicates(inplace=True)
n_after = len(df)
print(f"Duplicates removed: {n_before - n_after}")

# Check missing data percentage BEFORE filling
print("--- Missing Data Percentage per Column ---")
missing_pct = (df.isnull().mean() * 100).round(2)
print(missing_pct)
print("-" * 40)

# Impute any remaining missing values with column median
df.fillna(df.median(numeric_only=True), inplace=True)

# Final Checks
print(f"Remaining nulls: {df.isnull().sum().sum()}")
print(f"Final Shape: {df.shape}")
print(f"Time range: {df['Time'].min()} sec → {df['Time'].max()} sec")

# Display clean descriptive statistics
print("\n--- Cleaned Data Summary ---")
print(df[["Time", "Amount", "Class"]].describe().round(2).T)


#######################################################
#### M2 EXPLORATORY DATA ANALYSIS & VISUALIZATIONS ####
#######################################################

# Target key columns for EDA
cols = ["V1", "V2", "V3", "V4", "Amount", "Class"]

# ----------------------------------------------------------
# 1. Summary Statistics Output
# ----------------------------------------------------------
print("--- Clean Summary Statistics ---")
print(df[cols].describe().round(2).T)
print("\nGenerating charts on screen...")

# ----------------------------------------------------------
# 2. Variable Distributions & Skewness Check Matrix
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for ax, col in zip(axes, cols):
    ax.hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.7)
    ax.set_title(f"Distribution of {col}", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 3. Class Imbalance: Fraud vs Legitimate Transactions
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
class_counts = df["Class"].value_counts()
bars = ax.bar(["Legitimate (0)", "Fraud (1)"], class_counts.values,
              color=["steelblue", "crimson"], edgecolor="white", width=0.5)
for bar, count in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{count:,}\n({count/len(df)*100:.2f}%)", ha="center", fontsize=11)
ax.set_title("Class Distribution: Legitimate vs Fraudulent Transactions", fontsize=14)
ax.set_ylabel("Number of Transactions", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 4. Transaction Amount by Class (Box Plot Comparison)
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
legit_amounts = df[df["Class"] == 0]["Amount"]
fraud_amounts = df[df["Class"] == 1]["Amount"]
ax.boxplot([legit_amounts, fraud_amounts],
           labels=["Legitimate", "Fraud"],
           patch_artist=True,
           boxprops=dict(facecolor="steelblue", alpha=0.6),
           medianprops=dict(color="red", linewidth=2))
ax.set_title("Transaction Amount Distribution: Legitimate vs Fraud", fontsize=14)
ax.set_ylabel("Transaction Amount (€)", fontsize=12)
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 5. Annotated Correlation Heatmap Matrix
# ----------------------------------------------------------
heatmap_cols = ["V1", "V2", "V3", "V4", "V14", "V17", "Amount", "Class"]
corr = df[heatmap_cols].corr().round(2)
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(im, ax=ax)
ax.set_xticks(range(len(heatmap_cols)))
ax.set_yticks(range(len(heatmap_cols)))
ax.set_xticklabels(heatmap_cols, rotation=45, ha="right")
ax.set_yticklabels(heatmap_cols)
for i in range(len(heatmap_cols)):
    for j in range(len(heatmap_cols)):
        text_color = "white" if abs(corr.iloc[i, j]) > 0.5 else "black"
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                color=text_color, fontsize=9)
ax.set_title("Correlation Matrix (Key Features & Class)", fontsize=14)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 6. Hourly Transaction Volume Over Time (Dual Axis)
# ----------------------------------------------------------
# Convert Time (seconds) into hours
df["Hour"] = (df["Time"] // 3600).astype(int)
hourly_legit = df[df["Class"] == 0].groupby("Hour")["Amount"].mean()
hourly_fraud  = df[df["Class"] == 1].groupby("Hour")["Amount"].mean()

fig, ax1 = plt.subplots(figsize=(12, 6))
line1, = ax1.plot(hourly_legit.index, hourly_legit.values,
                  color="navy", linewidth=2, label="Avg Legit Amount (€)")
ax1.set_xlabel("Hour Since Recording Began", fontsize=12)
ax1.set_ylabel("Avg Legit Transaction Amount (€)", color="navy", fontsize=12)
ax1.tick_params(axis="y", labelcolor="navy")
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
line2, = ax2.plot(hourly_fraud.index, hourly_fraud.values,
                  color="crimson", linewidth=2, alpha=0.8, label="Avg Fraud Amount (€)")
ax2.set_ylabel("Avg Fraud Transaction Amount (€)", color="crimson", fontsize=12)
ax2.tick_params(axis="y", labelcolor="crimson")
ax2.grid(False)

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", frameon=True, fontsize=10)
plt.title("Hourly Average Transaction Amount: Legitimate vs Fraud (Dual Axis)", fontsize=14)
plt.tight_layout()
plt.show()


##########
####M3####
##########

x = df[df["Amount"] > 0]["Amount"].dropna()

# 1. Distribution Plot
plt.figure(figsize=(10, 4))
plt.hist(x, bins=60, edgecolor="black", color="skyblue", alpha=0.8)
plt.axvline(x.mean(),   color="red",   linestyle="--", linewidth=2, label=f"Mean: {x.mean():.2f}")
plt.axvline(x.median(), color="green", linestyle="-.", linewidth=2, label=f"Median: {x.median():.2f}")
plt.title("Distribution of Transaction Amount", fontsize=13)
plt.xlabel("Amount (€)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 2. Descriptive Statistics
print(" Descriptive Statistics (Transaction Amount) ")
print(f" Sample Size : {len(x)}")
print(f" Mean        : {x.mean():.3f} €")
print(f" Median      : {x.median():.3f} €")
print(f" Std Dev     : {x.std():.3f}")
print(f" Skewness    : {x.skew():.3f}")
print(f" Kurtosis    : {x.kurt():.3f}")
print()

# 3. Exceedance Probabilities
print(" Exceedance Probabilities (Amount vs Benchmarks) ")
thresholds = [100, 500, 1000, 5000]
for t in thresholds:
    prob = (x > t).mean()
    print(f" P(Amount > {t} €) = {prob:.4f} ({prob*100:.2f}%)")


######################
#### M4 Inference ####
######################

inf_data = df.dropna(subset=["Amount", "V14"])

# Pearson correlation between Amount and V14
# (V14 has one of the strongest correlations with fraud/Class)
correlation = inf_data["Amount"].corr(inf_data["V14"], method="pearson")
print(" M4: Correlation Inference ")
print(f"Pearson Correlation between Amount and V14: {correlation:.4f}")

if abs(correlation) > 0.7:
    strength = "Strong Positive" if correlation > 0 else "Strong Negative"
elif abs(correlation) > 0.4:
    strength = "Moderate Positive" if correlation > 0 else "Moderate Negative"
else:
    strength = "Weak"

print(f"Conclusion: There is a {strength} correlation.")
print("This suggests transaction amount alone is not a strong predictor of fraudulent behaviour.")

# Bonus: Correlation between V17 and Class
corr_v17 = df["V17"].corr(df["Class"], method="pearson")
print(f"\nPearson Correlation between V17 and Class (Fraud): {corr_v17:.4f}")
if abs(corr_v17) > 0.3:
    print("V17 shows a meaningful linear relationship with fraud classification.")


#######################
#### M5 Regression ####
#######################

# Filter valid rows
reg_data = pd.DataFrame({"Amount": df["Amount"], "Time": df["Time"]})
reg_data = reg_data[(reg_data["Amount"] > 0)].dropna()

X = reg_data["Time"]
y = reg_data["Amount"]

# Simple linear regression using scipy
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
r_squared = r_value ** 2

print(" Linear Regression: Predicting Transaction Amount based on Time ")
print(f"Equation  : Amount = {slope:.6f} * Time + {intercept:.4f}")
print(f"R-squared : {r_squared:.4f}")
print(f"p-value   : {p_value:.4e}")

if p_value < 0.05:
    print("Result: Statistically significant relationship (p < 0.05).")
else:
    print("Result: No statistically significant linear relationship between Time and Amount.")


##########################
#### M6 Visualization ####
##########################

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Linear Regression Line (Time vs Amount)
sample = reg_data.sample(n=5000, random_state=42)  # Sample for clarity
axes[0].scatter(sample["Time"], sample["Amount"], alpha=0.1, color="gray", label="Data Points")
x_vals = np.linspace(X.min(), X.max(), 100)
y_vals = slope * x_vals + intercept
axes[0].plot(x_vals, y_vals, color="red", linewidth=2,
             label=f"Fit: y={slope:.5f}x+{intercept:.2f}")
axes[0].set_title("Effect of Time on Transaction Amount")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Transaction Amount (€)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Plot 2: Hourly Average Transaction Count Profile
df_clean = df[df["Amount"] > 0].copy()
df_clean["Hour"] = (df_clean["Time"] // 3600).astype(int)
hourly_profile = df_clean.groupby("Hour")["Amount"].mean()
axes[1].plot(hourly_profile.index, hourly_profile.values,
             marker="o", color="teal", linewidth=2)
axes[1].set_title("Average Transaction Amount by Hour")
axes[1].set_xlabel("Hour Since Recording Began")
axes[1].set_ylabel("Mean Transaction Amount (€)")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
