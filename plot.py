from matplotlib import pyplot as plt
import pickle

with open("results/losses_50_relu_11", 'rb') as f:
    l_relu_11 = pickle.load(f)
with open("results/losses_50_relu_19", 'rb') as f:
    l_relu_19 = pickle.load(f)
with open("results/losses_50_prelu_11", 'rb') as f:
    l_prelu_11 = pickle.load(f)
with open("results/losses_50_prelu_19", 'rb') as f:
    l_prelu_19 = pickle.load(f)
with open("results/losses_linear_5_19", 'rb') as f:
    l_linear_prelu_19 = pickle.load(f)
with open("results/losses_decay_linear_19", 'rb') as f:
    l_decay_linear_prelu_19 = pickle.load(f)

#%%

length = 20
xs = range(length)
loss = 1
loss_type = 'valLoss' if loss == 1 else 'trainLoss'

fig, ax = plt.subplots()
ax.plot(xs, l_relu_11[loss_type][:length], '-', label='11 layers ReLU', color='red')
ax.plot(xs, l_relu_19[loss_type][:length], '-', label='19 layers ReLU', color='blue')
ax.plot(xs, l_prelu_11[loss_type][:length], '--', label='11 layers PReLU', color='red')
ax.plot(xs, l_prelu_19[loss_type][:length], '--', label='19 layers PReLU', color='blue')
ax.plot(xs, l_prelu_19[loss_type][:length], '--', label='19 layers PReLU', color='blue')

ax.legend()
fig.show()

#%%

length = 20
xs = range(length)
loss = 1
loss_type = 'valLoss' if loss == 1 else 'trainLoss'
loss_type = 'trainLoss'

fig, ax = plt.subplots()
ax.plot(xs, l_relu_19[loss_type][:length], '-', label='PReLU', color='red')
ax.plot(xs, l_linear_prelu_19[loss_type][:length], '-', label='Scheduled parameters', color='green')
ax.plot(xs, l_decay_linear_prelu_19[loss_type][:length], '-', label='Scheduled weight decay', color='blue')

loss_type = 'valLoss'

ax.plot(xs, l_relu_19[loss_type][:length], '--', color='red')
ax.plot(xs, l_linear_prelu_19[loss_type][:length], '--', color='green')
ax.plot(xs, l_decay_linear_prelu_19[loss_type][:length], '--', color='blue')


ax.legend()
fig.show()
