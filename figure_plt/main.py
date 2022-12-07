import matplotlib.pyplot as plt

models = ['FTT_light', 'FTT_heavy', 'Fastformer_light', 'Fastformer_heavy',
          'FTT_PFT_light', 'FTT_PFT_heavy', 'FTT_PT_heavy', 'FTT_PT_light', 'Fastformer_PFT_light', 'Fastformer_PFT_heavy']
train_time = [1., 3.95697362, 0.97865988, 4.32486879, 0.9974759, 3.55568501,
3.83584634, 1.00081175, 0.98002947, 3.61760785]

rank = [7.21904762, 3.10714286, 9.31904762, 4.12857143, 6.53095238, 3.37857143,
2.9, 6.42619048, 7.90714286, 4.08333333]


import numpy as np
ave = []
fs = 12
x = np.linspace(0, 3)
for i, m in enumerate(models):
    x, y = train_time[i], rank[i]
    plt.scatter(x, y, s=100)
    if x < 3:
        if m == "FTT_PT_light":
            plt.text(x + 0.1, y - 0.4, m, fontsize=fs)
        else:
            plt.text(x+0.15, y-0.13, m, fontsize=fs)
    else:
        if m=="Fastformer_PFT_heavy":
            plt.text(x-1.5, y, m, fontsize=fs)
        elif m=="FTT_PFT_heavy":
            plt.text(x-1.2, y-0.05, m, fontsize=fs)
        elif m=="FTT_PT_heavy":
            plt.text(x-1.1, y-0.05, m, fontsize=fs)
        elif m=="FTT_heavy":
            plt.text(x+0.1, y, m, fontsize=fs)
        else:
            plt.text(x - 1, y + 0.3, m, fontsize=fs)



ax = plt.gca()
plt.xlabel("Normalized train time")
plt.ylabel("Average model rank")
plt.savefig('foo.png', bbox_inches='tight')




