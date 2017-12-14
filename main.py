import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x_labels = ['soleado', 'nublado', 'lluvia']
colors = ['yellow', 'gray', 'blue']
data = np.loadtxt('tiempos.txt', delimiter=' ', skiprows=1, usecols=[1,2,3])
clean_data = np.loadtxt('tiempos_limpios.txt', delimiter=' ', skiprows=1, usecols=[1,2,3]) # limpiados a mano

transpose = clean_data.transpose()

def plot_times(data):
  x = [0, 1, 2]
  plt.xticks(x, x_labels)
  for row in data:
    plt.plot(x, row)

  plt.xlabel('clima')
  plt.ylabel('tiempo')
  plt.title('Variacion de tiempo segun el clima')
  plt.grid(True)
  plt.show()
  plt.clf()

def plot_times_bar(data):
  transpose = data.transpose()

  fig, ax = plt.subplots()

  index = np.arange(len(data))
  bar_width = 0.30

  opacity = 0.4

  for i, weather, color in zip(range(3), x_labels, colors):
    rects1 = plt.bar(index + bar_width * i, transpose[i], bar_width,
                     alpha=opacity,
                     color=color,
                     label=weather)

  plt.xlabel('Corredor')
  plt.ylabel('Tiempo')
  plt.title('Tiempos')
  plt.xticks(index + bar_width, range(1, 13))
  plt.legend()

  plt.tight_layout()
  plt.show()

def plot_avg_std(data):
  # rectangular box plot
  boxplot = plt.boxplot(data,
                       vert=True,   # vertical box aligmnent
                       patch_artist=True,
                       labels=x_labels)   # fill with color

  # fill with colors
  for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

  plt.grid(True)
  plt.xlabel('tiempo')
  plt.ylabel('clima')
  plt.show()

def plot_avg_std_diff(data):
  diffs = np.array([[row[0] - row[1], row[1] - row[2], row[0] - row[2]] for row in data])
  boxplot = plt.boxplot(diffs,
                       vert=True,   # vertical box aligmnent
                       patch_artist=True,
                       labels=['sol - nublado', 'nublado - lluvia', 'sol - lluvia'])   # fill with color

  # fill with colors
  colors_diff = ['yellow', 'gray', 'blue']
  for patch, color in zip(boxplot['boxes'], colors_diff):
    patch.set_facecolor(color)

  plt.grid(True)
  plt.xlabel('diferencia de tiempo')
  plt.ylabel('climas')
  plt.show()

def pearson_correlation(data):
  transpose = data.transpose()
  sol_vs_nub = stats.pearsonr(transpose[0], transpose[1])
  nub_vs_llu = stats.pearsonr(transpose[1], transpose[2])
  sol_vs_llu = stats.pearsonr(transpose[0], transpose[2])
  print('Coeficiente pearson Sol vs Nublado: {} | pvalue={}'.format(sol_vs_nub[0] ,sol_vs_nub[1]))
  print('Coeficiente pearson Nublado vs Lluvia: {} | pvalue={}'.format(nub_vs_llu[0] ,nub_vs_llu[1]))
  print('Coeficiente pearson Sol vs Lluvia: {} | pvalue={}'.format(sol_vs_llu[0] ,sol_vs_llu[1]))

def variance_eq_test(g1, g2):
  alpha = 0.05
  p_value = stats.f.cdf(np.var(g1)/np.var(g2), len(g1)-1, len(g2)-1)
  if stats.levene(g1,g2)[1] > alpha:
      return True

def normality_test(g):
  alpha = 0.05
  res = stats.shapiro(g)[1]
  return res >= alpha

def ttest(d1, d2):
  if normality_test(d1) and normality_test(d2):
    print("Distributions pass normality test")
    if variance_eq_test(d1,d2):
      print("Distributions seem to have equal variance")
      result = stats.ttest_rel(d1, d2)
      print('Student t-test apareado: pvalue={}'.format(result.pvalue))
    else:
      print("Distributions dont seem to have equal variance")
      result = stats.ttest_ind(d1, d2)
      print('Welch t-test independent: pvalue={}'.format(result.pvalue))
    if result.pvalue > 0.05:
      print('Averages scores are identical')
    else:
      print('Averages scores differ significantly')
  else:
    print("At least one of the distributions is not normal")
  return

def wilcoxon(data):
  transpose = data.transpose()
  sol_vs_nub = stats.wilcoxon(transpose[0], transpose[1])
  nub_vs_llu = stats.wilcoxon(transpose[1], transpose[2])
  sol_vs_llu = stats.wilcoxon(transpose[0], transpose[2])
  print('Wilcoxon rank Sol vs Nublado: {} | pvalue={}'.format(sol_vs_nub.statistic ,sol_vs_nub.pvalue))
  print('Wilcoxon rank Nublado vs Lluvia: {} | pvalue={}'.format(nub_vs_llu.statistic ,nub_vs_llu.pvalue))
  print('Wilcoxon rank Sol vs Lluvia: {} | pvalue={}'.format(sol_vs_llu.statistic ,sol_vs_llu.pvalue))

def exact_mc_perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

def difference_permutation_test(data):
  transpose = data.transpose()
  sol_vs_llu = transpose[0] - transpose[2]
  initial_mean = abs(transpose[0].mean() - transpose[2].mean())
  print('Initial mean: {}'.format(initial_mean))
  nreps = 1000
  resamp_means = np.empty(nreps)
  llu = np.array(transpose[2])

  for i in range(nreps):
    np.random.shuffle(llu)
    resamp = transpose[0] - llu
    resamp_means[i] = resamp.mean()

  print(resamp_means)
  highprob = len([x for x in resamp_means if x >= initial_mean])/nreps
  lowprob = len([x for x in resamp_means if x <= -1*initial_mean])/nreps
  print('high tail: {} | low tail: {}'.format(highprob, lowprob))
  prob2tailed = lowprob + highprob
  print("The probability from the sampling statistics is = ", prob2tailed)
  plt.hist(resamp_means, 15, normed=1, facecolor='green', alpha=0.75)
  plt.axvline(x=initial_mean, color='r')
  plt.ylabel('Density')
  plt.xlabel('Mean difference')
  plt.grid(True)
  plt.show()

def permutation_test(data):
  transpose = data.transpose()
  sol_vs_llu = transpose[0] - transpose[2]
  initial_mean = abs(transpose[0].mean() - transpose[2].mean())
  print('Initial mean: {}'.format(initial_mean))
  nreps = 1000
  resamp_means = np.empty(nreps)

  for i in range(nreps):
    signs = np.random.choice([-1,1], len(sol_vs_llu))
    resamp = sol_vs_llu * signs
    resamp_means[i] = resamp.mean()

  highprob = len([x for x in resamp_means if x >= initial_mean])/nreps
  lowprob = len([x for x in resamp_means if x <= -1*initial_mean])/nreps
  print('high tail: {} | low tail: {}'.format(highprob, lowprob))
  prob2tailed = lowprob + highprob
  print("The probability from the sampling statistics is = ", prob2tailed)
  plt.hist(resamp_means, 15, normed=1, facecolor='green', alpha=0.75)
  plt.axvline(x=initial_mean, color='r')
  plt.ylabel('Density')
  plt.xlabel('Mean difference')
  plt.grid(True)
  plt.show()

def run_1sample(group):
  stat, pvalue = stats.ttest_1samp(group, np.mean(group))
  print('T-test: {}| Pvalue: {}'.format(stat, pvalue))

def run_ttests():
  print('Test sol vs nublado: ')
  ttest(transpose[0], transpose[1])
  print('Test nublado vs lluvia: ')
  ttest(transpose[1], transpose[2])
  print('Test sol vs lluvia: ')
  ttest(transpose[0], transpose[2])

# plot_times(data)
# plot_times(clean_data)
# plot_avg_std(clean_data)
# run_ttests()
# pearson_correlation(clean_data)
# wilcoxon(clean_data)
#permutation_test(clean_data)
# run_1sample(transpose[2])
difference_permutation_test(clean_data)
