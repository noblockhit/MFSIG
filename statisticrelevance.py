from scipy import stats


g1 = [332, 328]
g2 = [293, 311, 304, 325]


# get the p-value

statistic, p = stats.ttest_ind(g1, g2, equal_var=False)

if p > 0.05:
    print("The difference between the groups is not statistically significant", end=" ")
else:
    print("The difference between the groups is statistically significant", end=" ")

print(f"(p-value: {p:.4f})")
