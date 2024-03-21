# import matplotlib.pyplot as plt

# # Data from the provided table
# data_ls = {
#     'Regularization ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     'dev-clean': [3.05, 2.92, 2.95, 2.89, 2.93, 2.91, 2.96, 2.97, 3.04, 3.18],
#     'dev-other': [7.78, 7.57, 7.57, 7.63, 7.52, 7.54, 7.65, 7.71, 7.97, 8.38],
#     'test-clean': [3.19, 3.16, 3.15, 3.13, 3.15, 3.14, 3.18, 3.14, 3.19, 3.45],
#     'test-other': [7.41, 7.37, 7.34, 7.36, 7.34, 7.32, 7.51, 7.47, 7.79, 8.27],
#     'avg': [5.36, 5.26, 5.25, 5.25, 5.24, 5.23, 5.33, 5.32, 5.50, 5.82]  # Calculated average
# }
# column_1 = [6.76, 6.97, 6.99, 6.99, 6.95, 7.04, 7.20, 7.16, 7.74, 7.97]
# column_2 = [13.15, 13.52, 13.24, 13.33, 13.62, 13.72, 13.95, 14.08, 14.66, 15.37]
# column_3 = [7.05, 7.24, 7.29, 7.23, 7.31, 7.29, 7.52, 7.58, 7.97, 8.41]
# column_4 = [14.20, 14.70, 14.27, 14.68, 14.71, 14.69, 15.14, 15.22, 15.87, 16.77]

# # Creating dictionary as requested
# data_l2 = {
#     'Regularization ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     'dev-clean': column_1,
#     'dev-other': column_2,
#     'test-clean': column_3,
#     'test-other': column_4,
# }

# # Plotting subplots for 'clean' and 'other'
# fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

# # Selecting colors for 'dev' and 'test' metrics
# dev_colors = '#00B0F0'
# test_colors = '#E32636'

# # Plotting for 'dev' metrics
# for idx, metric in enumerate(data_ls.keys()):
#     if 'test-clean' in metric or 'test-other' in metric and metric != 'Regularization ratio':
#         if metric == 'test-clean':
#             axs.plot(data_ls['Regularization ratio'], data_ls[metric], label=metric, color=dev_colors, marker='o', linewidth=2)
#         else:
#             axs.plot(data_ls['Regularization ratio'], data_ls[metric], label=metric, color=test_colors, marker='x', linewidth=2)

# axs.set_xlabel('Regularization ratio', fontsize=14)
# axs.set_ylabel('WER (%)', fontsize=14)
# axs.legend(fontsize=12)
# axs.grid(True)

# # Plotting for 'test' metrics
# for idx, metric in enumerate(data_l2.keys()):
#     if 'test-clean' in metric or 'test-other' in metric and metric != 'Regularization ratio':
#         if metric == 'test-clean':
#             axs.plot(data_l2['Regularization ratio'], data_l2[metric], label=metric, color=dev_colors, marker='o', linewidth=2)
#         else:
#             axs.plot(data_l2['Regularization ratio'], data_l2[metric], label=metric, color=test_colors, marker='x', linewidth=2)

# axs.set_xlabel('Regularization ratio', fontsize=14)
# axs.set_ylabel('WER (%)', fontsize=14)
# axs.legend(fontsize=12)
# axs.grid(True)

# # Display the plot
# plt.savefig('exp/plot_pngs/TH_vs_WER.png')

import matplotlib.pyplot as plt

# Data from the provided tables
data_ls = {
    'Regularization ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'dev-clean': [3.05, 2.92, 2.95, 2.89, 2.93, 2.91, 2.96, 2.97, 3.04, 3.18],
    'dev-other': [7.78, 7.57, 7.57, 7.63, 7.52, 7.54, 7.65, 7.71, 7.97, 8.38],
    'test-clean': [3.19, 3.16, 3.15, 3.13, 3.15, 3.14, 3.18, 3.14, 3.19, 3.45],
    'test-other': [7.41, 7.37, 7.34, 7.36, 7.34, 7.32, 7.51, 7.47, 7.79, 8.27],
}

data_l2 = {
    'Regularization ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'dev-clean': [6.76, 6.97, 6.99, 6.99, 6.95, 7.04, 7.20, 7.16, 7.74, 7.97],
    'dev-other': [13.15, 13.52, 13.24, 13.33, 13.62, 13.72, 13.95, 14.08, 14.66, 15.37],
    'test-clean': [7.05, 7.24, 7.29, 7.23, 7.31, 7.29, 7.52, 7.58, 7.97, 8.41],
    'test-other': [14.20, 14.70, 14.27, 14.68, 14.71, 14.69, 15.14, 15.22, 15.87, 16.77],
}

# Updating the dictionaries to reverse the order of 'Regularization ratio' and corresponding metrics
for ii, data in enumerate([data_ls, data_l2]):
    data['Regularization ratio'] = list(reversed(data['Regularization ratio']))
    for key in data:
        data[key] = list(reversed(data[key]))
    if ii == 0:
        data_ls = data
    else:
        data_l2 = data

# Setup for the 4 subplots
fig, axs = plt.subplots(1, 1, figsize=(6,5), sharey=True)

# Color and marker settings
greenish = '#006400'  # For dev sets
orangish = '#ff8c00'  # For test sets
# dark_green = '#006400'  # Darker green
# dark_orange = '#ff8c00'  # Darker orange
dev_marker = 'x'
test_marker = 'o'

# Plot for LS dev-clean and test-clean
axs.plot(data_ls['Regularization ratio'], data_ls['dev-clean'], label='dev-clean', color=greenish, marker=dev_marker, linewidth=2)
axs.plot(data_ls['Regularization ratio'], data_ls['test-clean'], label='test-clean', color=orangish, marker=test_marker, linewidth=2)
# axs.set_title('LS Clean')
axs.set_xlabel('Regularization ratio', fontsize=18)
axs.set_ylabel('WER (%)', fontsize=18)
axs.grid(True)
axs.legend(fontsize=14)
axs.tick_params(axis='x', labelsize=10)
axs.tick_params(axis='y', labelsize=10)
plt.savefig('exp/plot_pngs/TH_vs_WER1.png')
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6,5), sharey=True)
# Plot for LS dev-other and test-other
axs.plot(data_ls['Regularization ratio'], data_ls['dev-other'], label='dev-other', color=greenish, marker=dev_marker, linewidth=2)
axs.plot(data_ls['Regularization ratio'], data_ls['test-other'], label='test-other', color=orangish, marker=test_marker, linewidth=2)
# axs.set_title('LS Other')
axs.set_xlabel('Regularization ratio', fontsize=18)
axs.set_ylabel('WER (%)', fontsize=18)
axs.grid(True)
axs.legend(fontsize=14)
axs.tick_params(axis='x', labelsize=10)
axs.tick_params(axis='y', labelsize=10)
plt.savefig('exp/plot_pngs/TH_vs_WER2.png')
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6,5), sharey=True)
# Plot for L2 dev-clean and test-clean
axs.plot(data_l2['Regularization ratio'], data_l2['dev-clean'], label='dev-clean', color=greenish, marker=dev_marker, linewidth=2)
axs.plot(data_l2['Regularization ratio'], data_l2['test-clean'], label='test-clean', color=orangish, marker=test_marker, linewidth=2)
# axs.set_title('L2 Clean')
axs.set_xlabel('Regularization ratio', fontsize=18)
axs.set_ylabel('WER (%)', fontsize=18)
axs.grid(True)
axs.legend(fontsize=14)
axs.tick_params(axis='x', labelsize=10)
axs.tick_params(axis='y', labelsize=10)
plt.savefig('exp/plot_pngs/TH_vs_WER3.png')
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(6,5), sharey=True)
# Plot for L2 dev-other and test-other
axs.plot(data_l2['Regularization ratio'], data_l2['dev-other'], label='dev-other', color=greenish, marker=dev_marker, linewidth=2)
axs.plot(data_l2['Regularization ratio'], data_l2['test-other'], label='test-other', color=orangish, marker=test_marker, linewidth=2)
# axs.set_title('L2 Other')
axs.set_xlabel('Regularization ratio', fontsize=18)
axs.set_ylabel('WER (%)', fontsize=18)
axs.grid(True)
axs.legend(fontsize=14)
axs.tick_params(axis='x', labelsize=10)
axs.tick_params(axis='y', labelsize=10)
plt.savefig('exp/plot_pngs/TH_vs_WER4.png')
plt.close()