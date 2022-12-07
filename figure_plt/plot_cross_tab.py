import matplotlib.pyplot as plt

import numpy as np
x = np.linspace(0, 1000, 5)
time_tr = [1., 1.00024048, 0.99624789, 0.9955991, 0.99708937]
time_inf = [1., 1.00250918, 1.00480061, 1.00494291, 0.99813861]
perf = [3.41818182, 3.19545455, 2.95, 2.74318182, 2.69318182]
winrate = [0.5, 0.5911111111111111, 0.6238938053097347, 0.5964125560538118, 0.607929515418502]

ax = plt.subplot(4, 1, 1)
plt.plot(x, perf[:1]*5, '--', label='baseline')
plt.plot(x, perf, marker=".", markersize=10, label='using pretrained ckpt')
# ax.set_ylim([1, 5])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Model \n rank', fontsize=14)
ax.legend()

ax = plt.subplot(4, 1, 2, sharex=ax)
plt.plot(x, winrate[:1]*5, '--')
plt.plot(x, winrate, marker=".", markersize=10)
# ax.set_ylim([0.45, 1])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Win \n rate', fontsize=14)

ax = plt.subplot(4, 1, 3, sharex=ax)
plt.plot(x, time_tr[:1]*5, '--')
plt.plot(x, time_tr, marker=".", markersize=10)
ax.set_ylim([0.85, 1.05])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Finetune \n time', fontsize=14)

ax = plt.subplot(4, 1, 4, sharex=ax)
plt.plot(x, time_inf[:1]*5, '--')
plt.plot(x, time_inf, marker=".", markersize=10)
ax.set_ylim([0.9, 1.05])

ax.set_xlabel('Pretraining iterations', fontsize=14)
ax.set_ylabel('Test \n time', fontsize=14)

ax = plt.gca()
plt.savefig('light_FTTPreFT.png', bbox_inches='tight')


# heavy finetuning
plt.figure()
time_tr = [1., 0.99799819, 0.96567141, 0.9615077, 0.95088735]
time_inf = [1., 0.99658559, 0.99786463, 0.99592511, 0.99860741]
perf = [2.86342593, 2.90972222, 3.04398148, 3.20601852, 2.97685185]
winrate = [0.5, 0.46696035242290757, 0.46860986547085215, 0.4407894736842106, 0.4707207207207209]

ax = plt.subplot(4, 1, 1)
plt.plot(x, perf[:1]*5, '--', label='baseline')
plt.plot(x, perf, marker=".", markersize=10, label='using pretrained ckpt')
# ax.set_ylim([1, 5])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Model \n rank', fontsize=14)
ax.legend()

ax = plt.subplot(4, 1, 2, sharex=ax)
plt.plot(x, winrate[:1]*5, '--')
plt.plot(x, winrate, marker=".", markersize=10)
# ax.set_ylim([0.45, 1])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Win \n rate', fontsize=14)

ax = plt.subplot(4, 1, 3, sharex=ax)
plt.plot(x, time_tr[:1]*5, '--')
plt.plot(x, time_tr, marker=".", markersize=10)
ax.set_ylim([0.85, 1.05])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Finetune \n time', fontsize=14)

ax = plt.subplot(4, 1, 4, sharex=ax)
plt.plot(x, time_inf[:1]*5, '--')
plt.plot(x, time_inf, marker=".", markersize=10)
ax.set_ylim([0.9, 1.05])

ax.set_xlabel('Pretraining iterations', fontsize=14)
ax.set_ylabel('Test \n time', fontsize=14)

ax = plt.gca()
plt.savefig('heavy_FTTPreFT.png', bbox_inches='tight')


''' Pre-train '''
# light finetuning
plt.figure()
time_tr = [1., 0.99898979, 1.00100358, 0.99907505, 0.99646023]
time_inf = [1., 1.00233976, 1.00117145, 1.00233487, 0.99821735]
perf = [3.7037037, 3.37037037, 2.72453704, 2.58101852, 2.62037037]
winrate = [0.5, 0.6081081081081079, 0.6858407079646018, 0.7022222222222221, 0.6875]

ax = plt.subplot(4, 1, 1)
plt.plot(x, perf[:1]*5, '--', label='baseline')
plt.plot(x, perf, marker=".", markersize=10, label='using pretrained ckpt')
# ax.set_ylim([1, 5])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Model \n rank', fontsize=14)
ax.legend()

ax = plt.subplot(4, 1, 2, sharex=ax)
plt.plot(x, winrate[:1]*5, '--')
plt.plot(x, winrate, marker=".", markersize=10)
# ax.set_ylim([0.45, 1])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Win \n rate', fontsize=14)

ax = plt.subplot(4, 1, 3, sharex=ax)
plt.plot(x, time_tr[:1]*5, '--')
plt.plot(x, time_tr, marker=".", markersize=10)
ax.set_ylim([0.85, 1.05])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Finetune \n time', fontsize=14)

ax = plt.subplot(4, 1, 4, sharex=ax)
plt.plot(x, time_inf[:1]*5, '--')
plt.plot(x, time_inf, marker=".", markersize=10)
ax.set_ylim([0.9, 1.05])

ax.set_xlabel('Pretraining iterations', fontsize=14)
ax.set_ylabel('Test \n time', fontsize=14)

ax = plt.gca()
plt.savefig('light_FTTPreTr.png', bbox_inches='tight')

# heavy finetuning
plt.figure()
time_tr = [1., 1.01622409, 0.99212732, 1.01255844, 0.99419155]
time_inf = [1., 0.99531647, 0.99621266, 1.00117827, 0.99580959]
perf = [3.05530973, 3.07964602, 2.83185841, 3.03982301, 2.99336283]
winrate = [0.5, 0.49122807017543857, 0.5462555066079295, 0.5087719298245612, 0.5088105726872247]

ax = plt.subplot(4, 1, 1)
plt.plot(x, perf[:1]*5, '--', label='baseline')
plt.plot(x, perf, marker=".", markersize=10, label='using pretrained ckpt')
# ax.set_ylim([1, 5])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Model \n rank', fontsize=14)
ax.legend()

ax = plt.subplot(4, 1, 2, sharex=ax)
plt.plot(x, winrate[:1]*5, '--')
plt.plot(x, winrate, marker=".", markersize=10)
# ax.set_ylim([0.45, 1])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Win \n rate', fontsize=14)

ax = plt.subplot(4, 1, 3, sharex=ax)
plt.plot(x, time_tr[:1]*5, '--')
plt.plot(x, time_tr, marker=".", markersize=10)
ax.set_ylim([0.85, 1.05])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Finetune \n time', fontsize=14)

ax = plt.subplot(4, 1, 4, sharex=ax)
plt.plot(x, time_inf[:1]*5, '--')
plt.plot(x, time_inf, marker=".", markersize=10)
ax.set_ylim([0.9, 1.05])

ax.set_xlabel('Pretraining iterations', fontsize=14)
ax.set_ylabel('Test \n time', fontsize=14)

ax = plt.gca()
plt.savefig('heavy_FTTPreTr.png', bbox_inches='tight')

''' FastFTT Pre-FT '''
# light finetuning
plt.figure()
time_tr = [1., 1.0080114, 1.00458825, 1.00387064, 1.00412334]
time_inf = [1., 0.99681208, 0.99743256, 0.99723154, 0.99832143]
perf = [4.42386831, 3.30864198, 2.72427984, 2.39506173, 2.14814815]
winrate = [0.5, 0.7928286852589643, 0.873015873015873, 0.8700787401574805, 0.8844621513944224]

ax = plt.subplot(4, 1, 1)
plt.plot(x, perf[:1]*5, '--', label='baseline')
plt.plot(x, perf, marker=".", markersize=10, label='using pretrained ckpt')
# ax.set_ylim([1, 5])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Model \n rank', fontsize=14)
ax.legend()

ax = plt.subplot(4, 1, 2, sharex=ax)
plt.plot(x, winrate[:1]*5, '--')
plt.plot(x, winrate, marker=".", markersize=10)
# ax.set_ylim([0.45, 1])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Win \n rate', fontsize=14)

ax = plt.subplot(4, 1, 3, sharex=ax)
plt.plot(x, time_tr[:1]*5, '--')
plt.plot(x, time_tr, marker=".", markersize=10)
ax.set_ylim([0.85, 1.05])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Finetune \n time', fontsize=14)

ax = plt.subplot(4, 1, 4, sharex=ax)
plt.plot(x, time_inf[:1]*5, '--')
plt.plot(x, time_inf, marker=".", markersize=10)
ax.set_ylim([0.9, 1.05])

ax.set_xlabel('Pretraining iterations', fontsize=14)
ax.set_ylabel('Test \n time', fontsize=14)

ax = plt.gca()
plt.savefig('light_FastFTT.png', bbox_inches='tight')

# heavy finetuning
plt.figure()
time_tr = [1., 0.92897927, 0.89745337, 0.89631789, 0.88176937]
time_inf = [1., 1.00437359, 1.00450009, 1.00040788, 0.99880314]
perf = [3.31512605, 2.83193277, 2.94747899, 2.94327731, 2.96218487]
winrate = [0.5, 0.5737051792828685, 0.5918367346938775, 0.5582329317269077, 0.5582329317269077]

ax = plt.subplot(4, 1, 1)
plt.plot(x, perf[:1]*5, '--', label='baseline')
plt.plot(x, perf, marker=".", markersize=10, label='using pretrained ckpt')
# ax.set_ylim([1, 5])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Model \n rank', fontsize=14)
ax.legend()

ax = plt.subplot(4, 1, 2, sharex=ax)
plt.plot(x, winrate[:1]*5, '--')
plt.plot(x, winrate, marker=".", markersize=10)
# ax.set_ylim([0.45, 1])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Win \n rate', fontsize=14)

ax = plt.subplot(4, 1, 3, sharex=ax)
plt.plot(x, time_tr[:1]*5, '--')
plt.plot(x, time_tr, marker=".", markersize=10)
ax.set_ylim([0.85, 1.05])
plt.tick_params('x', labelbottom=False)
ax.set_ylabel('Finetune \n time', fontsize=14)

ax = plt.subplot(4, 1, 4, sharex=ax)
plt.plot(x, time_inf[:1]*5, '--')
plt.plot(x, time_inf, marker=".", markersize=10)
ax.set_ylim([0.9, 1.05])

ax.set_xlabel('Pretraining iterations', fontsize=14)
ax.set_ylabel('Test \n time', fontsize=14)

ax = plt.gca()
plt.savefig('heavy_FastFTT.png', bbox_inches='tight')