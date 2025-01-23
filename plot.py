from matplotlib import pyplot as plt
import numpy as np
import pickle

# Normal models
# ReLUs
with open("results/losses_50_relu_11", 'rb') as f:
    l_relu_11 = pickle.load(f)
with open("results/losses_50_relu_19", 'rb') as f:
    l_relu_19 = pickle.load(f)
    
# PReLUs
with open("results/losses_50_prelu_11", 'rb') as f:
    l_prelu_11 = pickle.load(f)
with open("results/losses_50_prelu_19", 'rb') as f:
    l_prelu_19 = pickle.load(f)
# PReLUs weights
with open("results/losses_19_p-weights", 'rb') as f:
    weights_19 = pickle.load(f)
with open("results/losses_11_p-weights", 'rb') as f:
    weights_11 = pickle.load(f)

# PReLU with linear weight scheduling
with open("results/losses_linear_5_19", 'rb') as f:
    l_linear_prelu_19 = pickle.load(f)

# PReLUs with linear scheduling on weight decay
with open("results/losses_decay_linear_01_19", 'rb') as f:
    l_decay_linear_prelu_01_19 = pickle.load(f)
with open("results/losses_decay_linear_04_19", 'rb') as f:
    l_decay_linear_prelu_04_19 = pickle.load(f)
with open("results/losses_decay_linear_19", 'rb') as f:
    l_decay_linear_prelu_2_19 = pickle.load(f)
# Their weights
with open("results/losses_decay_linear_01_19_p-weights", 'rb') as f:
    l_decay_linear_prelu_04_19_weights = pickle.load(f)
with open("results/losses_decay_linear_04_19_p-weights", 'rb') as f:
    l_decay_linear_prelu_04_19_weights = pickle.load(f)
with open("results/losses_decay_linear_19_p-weights", 'rb') as f:
    l_decay_linear_prelu_2_19_weights = pickle.load(f)

#%%

length = 20
ticks = np.arange(0, length, 3)
xs = range(length)
loss = 1
loss_type = 'valLoss' if loss == 1 else 'trainLoss'
loss_type = 'valLoss'

fig, ax = plt.subplots()
ax.plot(xs, l_relu_11[loss_type][:length], '--', color='red')
ax.plot(xs, l_prelu_11[loss_type][:length], '--', color='blue')
loss_type = 'trainLoss'
ax.plot(xs, l_relu_11[loss_type][:length], '-', label='ReLU', color='red')
ax.plot(xs, l_prelu_11[loss_type][:length], '-', label='PReLU', color='blue')

ax.set_xticks(ticks)
ax.set_ylim([0, 1.5])
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("11-layers model")
fig.savefig("../../proj_latex/images/relu_vs_prelu_11.png")

#%%

loss_type = 'valLoss'

fig, ax = plt.subplots()
ax.plot(xs, l_relu_19[loss_type][:length], '--', color='red')
ax.plot(xs, l_prelu_19[loss_type][:length], '--', color='blue')
loss_type = 'trainLoss'
ax.plot(xs, l_relu_19[loss_type][:length], '-', label='ReLU', color='red')
ax.plot(xs, l_prelu_19[loss_type][:length], '-', label='PReLU', color='blue')

ax.set_xticks(ticks)
ax.set_ylim([0, 1.5])
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("19-layers model")
fig.savefig("../../proj_latex/images/relu_vs_prelu_19.png")

#%%

length = 20
xs = range(length)
loss = 1
loss_type = 'valLoss' if loss == 1 else 'trainLoss'
loss_type = 'trainLoss'

fig, ax = plt.subplots()
ax.plot(xs, l_relu_19[loss_type][:length], '-', label='PReLU', color='red')
ax.plot(xs, l_linear_prelu_19[loss_type][:length], '-', label='Scheduled parameters', color='green')
ax.plot(xs, l_decay_linear_prelu_01_19[loss_type][:length], '-', label='Scheduled weak weight decay', color='blue')
ax.plot(xs, l_decay_linear_prelu_04_19[loss_type][:length], '-', label='Scheduled strong weight decay', color='orange')

loss_type = 'valLoss'

ax.plot(xs, l_relu_19[loss_type][:length], '--', color='red')
ax.plot(xs, l_linear_prelu_19[loss_type][:length], '--', color='green')
ax.plot(xs, l_decay_linear_prelu_01_19[loss_type][:length], '--', color='blue')
ax.plot(xs, l_decay_linear_prelu_04_19[loss_type][:length], '--', color='orange')


ax.legend()
fig.show()

#%%

weights_19_fixed = np.concatenate([[[1]*18], weights_19])
fig, ax = plt.subplots()
ax.plot(weights_19_fixed)
ax.set_xticks(ticks)
ax.set_ylim([-1, 4])
ax.set_xlim([0, 20])
ax.set_xlabel("Epochs")
ax.set_ylabel(r"$\alpha$ value")
ax.set_title("19-layers model")
plt.rcParams['text.usetex'] = True
fig.savefig("../../proj_latex/images/params_free_19.png")
plt.rcParams['text.usetex'] = False

#%%

weights_11_fixed = np.concatenate([[[1]*10], weights_11])
fig, ax = plt.subplots()
ax.plot(weights_11_fixed)
ax.set_xticks(ticks)
ax.set_ylim([-1, 4])
ax.set_xlim([0, 20])
ax.set_xlabel("Epochs")
ax.set_ylabel(r"$\alpha$ value")
ax.set_title("11-layers model")
plt.rcParams['text.usetex'] = True
fig.savefig("../../proj_latex/images/params_free_11.png")
plt.rcParams['text.usetex'] = False

#%%

weights_19_decay_04_fixed = np.concatenate([[[1]*18], l_decay_linear_prelu_04_19_weights])
fig, ax = plt.subplots()
ax.plot(weights_19_decay_04_fixed)
ax.set_xticks(ticks)
ax.set_xlim([0, 20])
ax.set_xlabel("Epochs")
ax.set_ylabel(r"$\alpha$ value")
ax.set_title("19-layers model")
plt.rcParams['text.usetex'] = True
fig.savefig("../../proj_latex/images/weights_decayed_04.png")
fig.show()
plt.rcParams['text.usetex'] = False

#%%

weights_19_decay_2_fixed = np.concatenate([[[1]*18], l_decay_linear_prelu_2_19_weights])
fig, ax = plt.subplots()
ax.plot(weights_19_decay_2_fixed)
ax.set_xticks(ticks)
ax.set_xlim([0, 20])
ax.set_xlabel("Epochs")
ax.set_ylabel(r"$\alpha$ value")
ax.set_title("19-layers model")
plt.rcParams['text.usetex'] = True
fig.savefig("../../proj_latex/images/weights_decayed_2.png")
fig.show()
plt.rcParams['text.usetex'] = False

#%%

length = 23
ticks = np.arange(0, length, 3)
xs = range(length)
loss = 1
loss_type = 'valLoss' if loss == 1 else 'trainLoss'
loss_type = 'valLoss'

fig, ax = plt.subplots()
ax.plot(xs, l_decay_linear_prelu_04_19[loss_type][:length], '--', color='blue')
ax.plot(xs, l_prelu_19[loss_type][:length], '--', color='red')
loss_type = 'trainLoss'
ax.plot(xs, l_decay_linear_prelu_04_19[loss_type][:length], '-', label='PReLU, with scheduled weight decay', color='blue')
ax.plot(xs, l_prelu_19[loss_type][:length], '-', label='Normal PReLU', color='red')

ax.plot([0, 50], [0, 0.4], '-.', color='grey', alpha=.5)

ax.set_xticks(ticks)
ax.set_ylim([0, 1.5])
ax.set_xlim([0, length-1])
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("19-layers model")
fig.show()
fig.savefig("../../proj_latex/images/prelu_vs_wdecay.png")

#%%

length = 23
ticks = np.arange(0, length, 3)
xs = range(length)
loss = 1
loss_type = 'valLoss' if loss == 1 else 'trainLoss'
loss_type = 'valLoss'

fig, ax = plt.subplots()
ax.plot(xs, l_linear_prelu_19[loss_type][:length], '--', color='blue')
ax.plot(xs, l_prelu_19[loss_type][:length], '--', color='red')
loss_type = 'trainLoss'
ax.plot(xs, l_linear_prelu_19[loss_type][:length], '-', label='PReLU, with scheduled shaping parameter', color='blue')
ax.plot(xs, l_prelu_19[loss_type][:length], '-', label='Normal PReLU', color='red')

ax.plot([0, 5], [1, 0], '-.', color='green', alpha=.5)

ax.set_xticks(ticks)
ax.set_ylim([0, 1.5])
ax.set_xlim([0, length-1])
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("19-layers model")
fig.show()
fig.savefig("../../proj_latex/images/prelu_vs_linear.png")

#%%

length = 23
ticks = np.arange(0, length, 3)
xs = range(length)
loss = 1
loss_type = 'valLoss' if loss == 1 else 'trainLoss'
loss_type = 'valLoss'

fig, ax = plt.subplots()
ax.plot(xs, l_decay_linear_prelu_2_19[loss_type][:length], '--', color='orange')
ax.plot(l_decay_linear_prelu_01_19[loss_type][:length], '--', color='green')
loss_type = 'trainLoss'
ax.plot(xs, l_linear_prelu_19[loss_type][:length], '-', label='PReLU, with scheduled fast parameter decay', color='orange')
ax.plot(l_decay_linear_prelu_01_19[loss_type][:length], '-', label='PReLU, with scheduled slow parameter decay', color='green')

ax.plot([0, 50], [0, 2], '-.', color='orange', alpha=.5)
ax.plot([0, 50], [0, .1], '-.', color='green', alpha=.5)

ax.set_xticks(ticks)
ax.set_ylim([0, 1.7])
ax.set_xlim([0, length-1])
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("19-layers model")
fig.show()
fig.savefig("../../proj_latex/images/fast_vs_slow_decay.png")
