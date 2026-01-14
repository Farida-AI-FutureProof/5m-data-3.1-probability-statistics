import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy import stats
import pandas as pd

# -----------------------------
# 1) Probability and Expected Value (3 coin flips)
# -----------------------------
np.random.seed(42)
samples = np.random.binomial(n=3, p=0.5, size=10000)

unique, counts = np.unique(samples, return_counts=True)
empirical_probs = counts / len(samples)
empirical_map = dict(zip(unique, empirical_probs))  # robust mapping

expected_value_emp = np.mean(samples)
theoretical_probs = [binom.pmf(k, n=3, p=0.5) for k in range(4)]
expected_value_theo = 3 * 0.5

print("=== Part 1: Coin flips ===")
print(f"Expected Value (Empirical):   {expected_value_emp:.4f}")
print(f"Expected Value (Theoretical): {expected_value_theo:.4f}\n")

print(f"{'Outcome':<10} {'Empirical':<15} {'Theoretical':<15} {'Difference':<15}")
print("-" * 55)
for outcome in range(4):
    emp = empirical_map.get(outcome, 0)
    theo = theoretical_probs[outcome]
    diff = emp - theo
    print(f"{outcome:<10} {emp:<15.4f} {theo:<15.4f} {diff:<15.4f}")

# Plot side-by-side bars
x = np.arange(4)
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, [empirical_map.get(k, 0) for k in x], width=bar_width, alpha=0.7, label="Empirical")
plt.bar(x + bar_width/2, theoretical_probs, width=bar_width, alpha=0.7, label="Theoretical")

plt.xlabel("Number of Heads")
plt.ylabel("Probability")
plt.title("Distribution of Coin Flips (3 coins, 10,000 samples)")
plt.xticks(x)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.show()


# -----------------------------
# 2) Normal Distribution and Statistical Testing
# -----------------------------
np.random.seed(42)
sample_a = np.random.normal(loc=70, scale=5, size=100)
sample_b = np.random.normal(loc=73, scale=5, size=100)

print("\n=== Part 2: Normal samples + t-test + CI ===")
print(f"Sample A - Mean: {sample_a.mean():.4f}, Std: {sample_a.std(ddof=1):.4f}")
print(f"Sample B - Mean: {sample_b.mean():.4f}, Std: {sample_b.std(ddof=1):.4f}")

# Box plot
plt.figure(figsize=(8, 5))
plt.boxplot([sample_a, sample_b], labels=["Sample A", "Sample B"])
plt.ylabel("Values")
plt.title("Box Plot Comparison")
plt.grid(axis="y", alpha=0.3)
plt.show()

# Welch's t-test (safer default)
t_statistic, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
print("\nIndependent t-test (Welch) Results:")
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value:     {p_value:.6f}")
print("Conclusion:", "Significantly different (p < 0.05)" if p_value < 0.05 else "NOT significantly different (p >= 0.05)")

# 95% confidence intervals for the mean
confidence_level = 0.95
se_a = stats.sem(sample_a)
se_b = stats.sem(sample_b)
ci_a = stats.t.interval(confidence_level, len(sample_a)-1, loc=sample_a.mean(), scale=se_a)
ci_b = stats.t.interval(confidence_level, len(sample_b)-1, loc=sample_b.mean(), scale=se_b)

print("\n95% Confidence Intervals (mean):")
print(f"Sample A: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
print(f"Sample B: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

# Visualize means + CIs (simple and clear)
plt.figure(figsize=(8, 5))
means = [sample_a.mean(), sample_b.mean()]
cis = [ci_a, ci_b]
xpos = [1, 2]

plt.scatter([1]*len(sample_a), sample_a, alpha=0.35)
plt.scatter([2]*len(sample_b), sample_b, alpha=0.35)

for i, (m, ci) in enumerate(zip(means, cis), start=1):
    plt.plot([i-0.15, i+0.15], [m, m], linewidth=3)
    plt.fill_between([i-0.15, i+0.15], ci[0], ci[1], alpha=0.25)

plt.xticks([1, 2], ["Sample A", "Sample B"])
plt.ylabel("Values")
plt.title("Distributions with 95% Confidence Intervals (Mean)")
plt.grid(axis="y", alpha=0.3)
plt.show()


# -----------------------------
# 3) Correlation Analysis (Iris)
# -----------------------------
print("\n=== Part 3: Correlation on Iris ===")

# Reliable iris load (no internet needed)
from sklearn.datasets import load_iris
iris_raw = load_iris(as_frame=True)
iris = iris_raw.frame.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
})

pairs = [
    ("sepal_length", "petal_length"),
    ("sepal_width", "petal_width")
]

results = {}

plt.figure(figsize=(14, 5))

for idx, (var1, var2) in enumerate(pairs, start=1):
    x = iris[var1]
    y = iris[var2]

    pearson_r, p_val = stats.pearsonr(x, y)
    r_squared = pearson_r ** 2

    results[f"{var1} vs {var2}"] = {"Pearson r": pearson_r, "p-value": p_val, "R-squared": r_squared}

    print(f"\n{var1.replace('_',' ').title()} vs {var2.replace('_',' ').title()}:")
    print(f"  Pearson r: {pearson_r:.4f}")
    print(f"  p-value:   {p_val:.2e}")
    print(f"  R-squared: {r_squared:.4f} ({r_squared*100:.2f}% variance explained)")

    plt.subplot(1, 2, idx)
    plt.scatter(x, y, alpha=0.6, s=40)

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_line, p(x_line), "r--", linewidth=2)

    plt.xlabel(var1.replace("_", " ").title())
    plt.ylabel(var2.replace("_", " ").title())
    plt.title(f"{var1.replace('_',' ').title()} vs {var2.replace('_',' ').title()}\n"
              f"r={pearson_r:.4f}, p={p_val:.2e}, R²={r_squared:.4f}")
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Compare
pair1 = "sepal_length vs petal_length"
pair2 = "sepal_width vs petal_width"

r2_1 = results[pair1]["R-squared"]
r2_2 = results[pair2]["R-squared"]

print("\n" + "="*60)
print("COMPARISON AND CONCLUSION")
print("="*60)
print(f"{pair1}: R²={r2_1:.4f}")
print(f"{pair2}: R²={r2_2:.4f}")

if r2_1 > r2_2:
    print(f"\nStronger relationship: {pair1} (higher R²)")
else:
    print(f"\nStronger relationship: {pair2} (higher R²)")

print("\nExplanation (plain English):")
print("- A higher |r| and higher R² means points lie closer to a straight line, so one variable predicts the other better.")
print("- In iris, petal measurements usually track species differences more strongly than sepal width does, so correlations differ in strength.")
