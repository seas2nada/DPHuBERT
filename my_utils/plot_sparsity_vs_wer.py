from matplotlib import pyplot as plt

p = [[2.71, 6.59, 2.90, 6.45], [3.56, 10.05, 3.87, 10.03], [3.61, 10.05, 3.87, 9.99], [3.74, 10.45, 3.93, 10.43], [3.82, 10.86, 3.99, 11.03], [3.92, 11.43, 4.21, 11.67], [4.27, 12.40, 4.58, 12.61], [4.80, 14.27, 4.91, 14.97], [5.81, 17.98, 6.08, 18.97], [9.75,28.97,10.00,30.22]]

regp = [[2.71, 6.59, 2.90, 6.45], [2.86, 7.13, 3.02, 7.13], [2.91, 7.54, 3.14, 7.32], [3.03, 8.21, 3.18, 7.81], [3.14, 8.89, 3.42, 8.51], [3.45, 10.03, 3.69, 9.86], [3.89, 11.15, 4.08, 11.41], [4.54, 13.25, 4.87, 13.76], [6.13,19.78,6.47,20.37], [12.70,34.91,12.93,36.61]]

omp = [[2.71, 6.59, 2.90, 6.45, 4.66],[2.79, 6.63, 2.92, 6.57, 4.73],[2.94, 7.26, 3.06, 7.39, 5.16],[4.26, 12.16, 4.57, 12.87, 8.47],[39.22, 63.98, 37.64, 66.71, 51.89],[99.84, 99.91, 99.79, 99.87, 99.85],[100, 100, 100, 100, 100],[100, 100, 100, 100, 100],[100, 100, 100, 100, 100], [100, 100, 100, 100, 100]]

sparsities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

base_model = (6.12+13.55+6.06+13.34) / 4

# Plotting subplots for 'clean' and 'other'
fig, axs = plt.subplots(1, 1, figsize=(13, 6), sharex=True)

# Selecting colors for 'dev' and 'test' metrics
regp_color = '#00B0F0'
p_color = '#E32636'
omp_color = '#FFA07A'

plot_omp = False

# Plotting for 'dev' metrics
new_p = []
new_regp = []
new_omp = []
for idx, sparsity in enumerate(sparsities):
    new_p_ = 0
    new_regp_ = 0
    new_omp_ = 0
    for p_, regp_, omp_ in zip(p[idx], regp[idx], omp[idx]):
        new_p_ += p_
        new_regp_ += regp_
        new_omp_ += omp_

    new_p.append(new_p_ / 4)
    new_regp.append(new_regp_ / 4)
    new_omp.append(new_omp_ / 5)

axs.plot(sparsities, new_regp, label="w/ GGPR", color='#1E88E5', marker='o', linestyle='-', linewidth=2)
axs.plot(sparsities, new_p, label="w/o GGPR", color="#D81B60", marker='x', linestyle='--', linewidth=2)
if plot_omp:
    axs.plot(sparsities, new_omp, label="OMP", color=omp_color, marker='s', linewidth=2)

# axs.scatter([0.7], [base_model], color='green', marker='*', s=100, label='wav2vec2-base-100h')

axs.legend()

axs.set_xlabel('Sparsity (%)', fontsize=22)
axs.set_ylabel('WER (%)', fontsize=22)
# axs.set_ylim(0, 10)
axs.tick_params(axis='x', labelsize=13)
axs.tick_params(axis='y', labelsize=13)
axs.legend(fontsize=18)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig('exp/plot_pngs/sparsity_vs_wer.png')
plt.close()