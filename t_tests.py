from scipy import stats
import numpy as np

import numpy as np

# Example population data (replace with your actual population data)
population_data = [10, 12, 15, 11, 13, 16, 14]

# Calculate the population mean
population_mean = np.mean(population_data)

print(f"The population mean is: {population_mean}")

# --- One-Sample T-test ---
print("--- One-Sample T-test ---")
# Example: Comparing the mean of a sample to a known population mean
sample = np.array([22, 25, 28, 23, 26, 27, 24, 25])
population_mean = 25
t_statistic, p_value = stats.ttest_1samp(sample, population_mean)
print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.3f}")
print()

# --- Independent (Two-Sample) T-test ---
print("--- Independent (Two-Sample) T-test ---")
# Example: Comparing the means of two independent samples
group1 = np.array([1, 2, 3, 4, 5])
group2 = np.array([2, 4, 4, 5, 6])
t_statistic_ind, p_value_ind = stats.ttest_ind(group1, group2)
print(f"T-statistic: {t_statistic_ind:.3f}")
print(f"P-value: {p_value_ind:.3f}")
print()

# If you assume equal variances (Welch's t-test if not)
t_statistic_ind_eq_var, p_value_ind_eq_var = stats.ttest_ind(group1, group2, equal_var=True)
print("Assuming equal variances:")
print(f"T-statistic: {t_statistic_ind_eq_var:.3f}")
print(f"P-value: {p_value_ind_eq_var:.3f}")
print()

# --- Paired-Sample T-test ---
print("--- Paired-Sample T-test ---")
# Example: Comparing the means of two related samples (e.g., before and after)
before = np.array([10, 12, 15, 11, 13])
after = np.array([12, 15, 16, 13, 14])
t_statistic_rel, p_value_rel = stats.ttest_rel(before, after)
print(f"T-statistic: {t_statistic_rel:.3f}")
print(f"P-value: {p_value_rel:.3f}")
