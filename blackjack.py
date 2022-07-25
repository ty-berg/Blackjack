from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def value_iteration(
    V0: npt.NDArray, 
    lr: float, 
    gamma:float, 
    epsilon: float=1e-12
    ) -> npt.NDArray:
  Vlast = V0
  i = 0
  while True:
      maxdiff = epsilon
      Vnew = np.zeros(V0.size)
      for x in range(V0.size):
        sum = 0
        for y in range(1,11):
          if x + y > 21:
            if y == 10:
              sum = sum + (4/13)*(lr)
            else:
              sum = sum + (1/13)*(lr)
          else:
            if y == 10:
              sum = sum + (4/13)*(lr + gamma*Vlast[x+y])
            else:
              sum = sum + (1/13)*(lr + gamma*Vlast[x+y])
        Vnew[x] = max(sum,x)
        if Vnew[x]-Vlast[x] > maxdiff:
          maxdiff = Vnew[x] - Vlast[x]
      for s in range(V0.size):
        Vlast[s] = Vnew[s]
      if maxdiff <= epsilon:
        break
  return Vlast
  
def value_to_policy(V: npt.NDArray, lr: float, gamma: float) -> npt.NDArray:
  Vpolicy = np.zeros(V.size)
  for x in range(V.size):
    if x == V[x]:
      Vpolicy[x] = 0
    else:
      Vpolicy[x] = 1
  return Vpolicy
  
def draw() -> int:
  probs = 1/13*np.ones(10)
  probs[-1] *= 4
  return np.random.choice(np.arange(1,11), p=probs)

def Qlearn(
    Q0: npt.NDArray, 
    lr: float, 
    gamma: float, 
    alpha: float, 
    epsilon: float, 
    N: int
    ) -> Tuple[npt.NDArray, npt.NDArray]:
  currS = 0
  record = np.zeros((N,3))
  for i in range(N):
    greedy = np.random.random()
    if greedy < epsilon:
      action = np.random.choice(2)
      if action == 0:
        Qval = Q0[currS,0]
        Qval = Qval + alpha*(currS - Qval)
        Q0[currS,0] = Qval
        record[i,0] = currS
        record[i,1] = 0
        record[i,2] = currS
        currS = 0
      else:
        Qval = Q0[currS,1]
        d = draw()
        succ = currS + d
        if succ <= 21 and Q0[succ, 0] < Q0[succ, 1]:
          max = Q0[succ,1]
        elif succ <= 21:
          max = Q0[succ,0]
        else:
          max = 0
        Qval = Qval + alpha*(lr + gamma*max - Qval)
        Q0[currS,1] = Qval
        record[i,0] = currS
        record[i,1] = 1
        record[i,2] = lr
        if succ > 21:
          currS = 0
        else:
          currS = succ
    else:
      if Q0[currS,0] < Q0[currS,1]:
        Qval = Q0[currS,1]
        d = draw()
        succ = currS + d
        if succ <= 21 and Q0[succ, 0] < Q0[succ, 1]:
          max = Q0[succ,1]
        elif succ <= 21:
          max = Q0[succ,0]
        else:
          max = 0
        Qval = Qval + alpha*(lr + gamma*max - Qval)
        Q0[currS,1] = Qval
        record[i,0] = currS
        record[i,1] = 1
        record[i,2] = lr
        if succ > 21:
          currS = 0
        else:
          currS = succ
      else:
        Qval = Q0[currS,0]
        Qval = Qval + alpha*(currS - Qval)
        Q0[currS,0] = Qval
        record[i,0] = currS
        record[i,1] = 0
        record[i,2] = currS
        currS = 0 
  return (Q0, record)

def RL_analysis():
  lr, gamma, alpha, epsilon, N = 0, 1, 0.1, 0.1, 10000
  visits = np.zeros((22,6))
  rewards = np.zeros((N,6))
  values = np.zeros((22,6))

  for i in range(6):
    _, record = Qlearn(np.zeros((22,2)), lr, gamma, alpha, epsilon, 10000*i)
    vals, counts = np.unique(record[:,0], return_counts=True)
    visits[vals.astype(int),i] = counts
    _, record = Qlearn(np.zeros((22,2)), lr, gamma, alpha, 0.2*i, N)
    rewards[:,i] = record[:,2]
    vals, _ = Qlearn(np.zeros((22,2)), lr, gamma, min(0.2*i+0.1,1), epsilon, N)
    values[:,i] = np.max(vals, axis=1)

  plt.figure()
  plt.plot(visits)
  plt.legend(['N=0', 'N=10k', 'N=20k', 'N=30k' ,'N=40k', 'N=50k'])
  plt.title('Number of visits to each state')

  plt.figure()
  plt.plot(np.cumsum(rewards, axis=0))
  plt.legend(['e=0.0', 'e=0.2', 'e=0.4' ,'e=0.6', 'e=0.8', 'e=1.0'])
  plt.title('Cumulative rewards received')

  plt.figure()
  plt.plot(values)
  plt.legend(['a=0.1' ,'a=0.3', 'a=0.5', 'a=0.7', 'a=0.9', 'a=1.0'])
  plt.title('Estimated state values');
