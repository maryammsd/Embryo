import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kendalltau,fisher_exact

def fisher_exact_test(child, parent):
    print("-----------------------------")
    print("-----------------------------")
    print("Performing Fisher's Exact Test")
    print("Child levels and counts vs Parent levels and counts:")
    print("-----------------------------")
    for level in child.keys():
        print(f"Level: {level}, Child count: {len(child[level])}, Parent count: {len(parent[level])}")
        data = np.array([[len(child[level]), len(parent[level])],
                         [sum(len(child[l]) for l in child if l != level),
                          sum(len(parent[l]) for l in parent if l != level)]])
        oddsratio, p_value = stats.fisher_exact(data)
        print(f"Level: {level}, Fisher's Exact Test: odds ratio = {oddsratio:.3f}, p-value = {p_value:.3f}")
def chi_test(child, parent):
    print("-----------------------------")
    print("-----------------------------")
    print("Performing Chi-squared Test")
    print("Child levels and counts vs Parent levels and counts:")
    print("-----------------------------")
    for level in child.keys():
        print(f"Level: {level}, Child count: {len(child[level])}, Parent count: {len(parent[level])}")
        data = np.array([child[level], parent[level]])
        chi2, p, dof, expected = stats.chi2_contingency(data)
        print(f"Level: {level}, Chi-squared Test: chi2 = {chi2:.3f}, p-value = {p:.3f}")
        print(f" Percentages of the cells with expected frequency < 5: {(expected < 5).mean() * 100:.2f}% ")


def spearmanr_test(child, parent):
    print("-----------------------------")
    print("-----------------------------")
    print("Performing Spearman Rank-Order Correlation Test")
    print("Child levels and counts vs Parent levels and counts:")
    print("-----------------------------")
    for level in child.keys():
        print(f"Level: {level}, Child count: {len(child[level])}, Parent count: {len(parent[level])}")
        correlation, p_value = spearmanr(child[level], parent[level])
        print(f"Level: {level}, Spearman Correlation: {correlation:.3f}, p-value = {p_value:.3f}")

def kendalltau_test(child, parent):
    print("-----------------------------")
    print("-----------------------------")
    print("Performing Kendall Tau Correlation Test")
    print("Child levels and counts vs Parent levels and counts:")
    print("-----------------------------")
    for level in child.keys():
        print(f"Level: {level}, Child count: {len(child[level])}, Parent count: {len(parent[level])}")
        correlation, p_value = kendalltau(child[level], parent[level])
        print(f"Level: {level}, Kendall Tau Correlation: {correlation:.3f}, p-value = {p_value:.3f}")

