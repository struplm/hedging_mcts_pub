# dh counterexample
# in each of "n_decisions" iteration, a shallow NN tryies to guess the right actions (Q is always bimodal),
# the MDP state is here given just by iteration index
# It can be easily observed that with "n_decisions" the probability of right guess in all iterations goes to 0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import norm
import matplotlib.pyplot as plt

import time
import os



n_decisions = 5
right_decisions = torch.tensor([0,1,0,1,0],dtype = torch.int64)
modeval = torch.tensor([200.0,100.0])


batch_size = 32
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_sim =  batch_size*1000
n_epochs = 10

enable_noise_input = True #extiting shalow nets with noise
enable_unbiased_modes = True # the modes are chosen symetricaly around 0 (unbiased), or in [0,1] interval (biased)

class JustBias(nn.Module):
  def __init__(self,in_features, out_features):
    self.biases = torch.zeros([out_features])

  def forward(self,x):
    return torch.zeros([x.shape[0],out_features])+self.biases



def get_rewards(actions,right_decision) :  
  epss = modeval**(-1)
  modearg = torch.zeros([len(actions),2])
  modearg[:,0] = right_decision
  modearg[:,1] = 1-right_decision
  modearg = modearg -0.5
  #modearg = torch.tensor([list(right_decision), list(1-right_decision)])-0.5
  #print(f"actions={actions.shape}, right_decision={right_decision.shape}, modearg={modearg.shape}")
  reward = (((actions[:,None]-modearg)**2+epss[None,:])**(-1)).sum(dim=1)
  return reward


# here is the simplest setup of a NN
# so the option pays at maturyti, T. The NN has now many layers, which respectively output the hedging decisions
# at the individual time steps
class DHlikeNN(nn.Module):
  def __init__(self, n_decisions):
    super(DHlikeNN, self).__init__()    
    self.contentNN = nn.ModuleList([nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 32), nn.Linear(32, 1)]) for i in range(n_decisions)])        
  def forward(self,x):
    assert x.shape[1] == n_decisions
    returns = torch.zeros([x.shape[0]])
    actions = torch.zeros(x.shape)
    for t, shallowNNatDt in enumerate(self.contentNN):
      input = x[:,t:t + 1 ] if enable_noise_input else torch.zeros(x[:,t:t + 1 ].shape,dtype = torch.double)
      innerVal = shallowNNatDt[0]( input )
      innerVal = F.tanh(innerVal)
      innerVal = shallowNNatDt[1](innerVal)
      innerVal = F.tanh(innerVal)
      actions[:,t] = shallowNNatDt[2](innerVal)[:,0]
      #returns = returns + get_rewards(actions[:,t],right_decisions[t]) # original
      returns = returns + get_rewards(actions[:,t],(x[:,t] > 0).int()) # quantization in reward function
      #returns = returns + get_rewards(actions[:,t],x[:,t])            # quantization in dataset
    self.returns = returns
    self.actions = actions
    return self.actions, self.returns


class MinimalDataset(torch.utils.data.Dataset):
  def __init__(self, n_sim, n_decisions):        
    self.data = np.random.randn(n_sim,n_decisions)
    #self.data = self.data > 0 # quantization in dataset
    self.data = self.data.astype(np.double)
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    retval = self.data[idx, :]        
    return retval







minimalDataset = MinimalDataset(n_sim=n_sim, n_decisions=n_decisions)

dataLoader = torch.utils.data.DataLoader(minimalDataset, batch_size = batch_size, shuffle=True) #32

dh_like_nn = DHlikeNN(n_decisions).double().to(device)

optimizer = optim.Adam(dh_like_nn.parameters(), lr=lr) # need to be switched!!! #hedger.parameters()

#torch.autograd.set_detect_anomaly(True)


start_time = time.time()
ireport = 100

loss_ar = np.zeros([n_epochs,len(dataLoader)])
actions_ar = np.zeros([n_epochs,len(dataLoader),batch_size,n_decisions])


for epoch in range(n_epochs):
  lossEpoch = 0
  for i, batch in enumerate(dataLoader):
    optimizer.zero_grad()

    actions, returns  = dh_like_nn (batch)      
    #assert False,"Konec"

    loss = -returns.mean()
            
    with torch.no_grad() :
      loss_ar[epoch,i] = loss.detach().cpu().numpy()
      actions_ar[epoch,i] = actions.detach().cpu().numpy()
      
      optimizer.zero_grad()    

      loss.backward()

      optimizer.step()

      lossEpoch += loss.item()

  print(f"Epoch : {epoch + 1}/{n_epochs}, i={i}, loss = {lossEpoch/i:.6f}, after secs = {time.time() - start_time},loss.dev={loss.device}")  



avgloss = lossEpoch/i

n_right_decisions = (-avgloss - n_decisions*modeval[1])/(modeval[0]-modeval[1])
n_right_decisions_err = n_right_decisions-np.round(n_right_decisions)
n_right_decisions = np.round(n_right_decisions).int()

print(f"sucess={n_right_decisions==n_decisions}, n_right_decisions={n_right_decisions}, n_right_decisions_err = {n_right_decisions_err}")

aN = 100
aR = np.linspace(-1,2,aN)
rev = np.zeros([aN])
t=0
for ai in range(aN) :
  with torch.no_grad() :
    rev[ai] = get_rewards( torch.tensor([aR[ai]]), right_decisions[t] ).cpu().numpy()

plt.figure()
plt.plot(aR,rev)

ysupp = np.outer(np.arange(0,n_epochs*len(dataLoader)),np.ones([batch_size*n_decisions])).reshape(-1)
plt.figure()
plt.scatter(ysupp,actions_ar.reshape(-1),s=0.2)

plt.figure()
xN = 100
xR = np.linspace(-1,1,xN)
xinp = xR[:,None]*np.ones([n_decisions])[None,:]
with torch.no_grad() :
  actions, returns  = dh_like_nn (torch.tensor(xinp))      
  for t in range(n_decisions) :
    plt.plot(xR,actions[:,t],label=f"t={t}")
plt.legend()
plt.xlabel("x - NN input")
plt.ylabel("action")

plt.show()
