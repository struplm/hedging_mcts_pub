# shared version with 2 modes and failing dep hedging

import numpy as np
import matplotlib.pyplot as plt

class Container :
  pass



# utility functions:
def quadratic_utility(self,x) :
  return -x**2

def quadratic_utility_clipped(self,x) :
  #tmp = -self.quadratic_utility(x)  
  tmp = x**2  
  tmp_scaled = (100/self.s0**2)*tmp
  tmp_clipped = np.minimum(20,tmp_scaled) 
  tmp_tr = -tmp_clipped/10+1 
  return tmp_tr

def log_utility(self,x) :
  # concave, increasing utility 
  # for negative values fixed with identity
  y = np.zeros(x.shape)
  y[x >= 0] = np.log(1+x[x >= 0])
  y[x < 0] = x[x < 0]
  return y

def exp_utility(self,x) :
  return -1/self.exp_ulility_alpha*np.exp(-self.exp_ulility_alpha*x)



# cost functions:
#ndiff = np.abs(n_ - n)
def cost_fixed(self,ndiff,s) :
  c = np.ones(ndiff.shape)*self.fixed_cost
  c[ndiff < self.c_treshold] = 0
  return c

def cost_2slopes(self,ndiff,s) :
  c = np.minimum(self.c_0*ndiff, self.c_1*ndiff+self.c_1s)
  return c

def cost_2slopes10(self,ndiff,s) :
  c = 10*np.minimum(self.c_0*ndiff, self.c_1*ndiff+self.c_1s)
  return c

def cost_2slopes50(self,ndiff,s) :
  c = 50*np.minimum(self.c_0*ndiff, self.c_1*ndiff+self.c_1s)
  return c



def cost_quadratic(self,ndiff,s) :
  c = self.c_quad_coef*(ndiff*s)**2
  return c




    # if env.cost == "fixed" :
    #   c = np.ones(n.shape)*env.fixed_cost
    #   c[np.abs(n_ - n) < env.c_treshold] = 0
    # if env.cost == "2slopes" :    
    #   ndiff = np.abs(n_ - n)
    #   c = np.minimum(env.c_0*ndiff, env.c_1*ndiff+env.c_1s)
    # if env.cost == "quadratic" :      
    #   c = env.c_quad_coef*((n_-n)*s)**2



def make_env_factory(utility_name, sN, sstep, p_up, p_down, aN, alim, T, isbounded, cost, cliped_reward, epsilon, bmax=None,bmin=None):

  #sN = 9 
  #sstep = 1 
  sR = np.linspace(sstep,sstep*sN,sN)
  s0 = sR[int(sN//2)]
  smax = np.max(sR)


  # trinomial model
  skernel2 = np.zeros([sN,sN])
  #p_up = 0.2
  #p_down = 0.2
  p_mid = 1-p_up-p_down
  skernel2 += np.eye(sN,sN,0)*p_mid
  skernel2 += np.eye(sN,sN,-1)*p_up
  skernel2 += np.eye(sN,sN,1)*p_down
  skernel2[0,0] += p_down
  skernel2[sN-1,sN-1] += p_up


  #aN = 20 
  #alim = 1 
  astep = alim/aN 

  #T = 5
   
  bmax = T/2*smax*alim if bmax is None else bmax
  bmin = -bmax if bmin is None else bmin #0 symetric (no bounds)

  #isbounded = False
  assert isbounded in set([True,False])

  assert cost in set(["fixed","2slopes","2slopes10", "2slopes50","quadratic" ])

  # prarameters of various cost functions are stored in env object (used for setting money discretization grid)
  fixed_cost = 0.05 
  c_treshold = 0.2
  c_0 = fixed_cost / c_treshold
  c_1 = 0.0 
  c_1s = fixed_cost - c_1*c_treshold
  c_quad_coef = 0.5


  if cost in set(["2slopes","2slopes10","2slopes50","fixed"]) :
    bstep = astep*sstep*0.5*0.5 # this assumes 2 slopes costs
    bN = int((bmax-bmin)//bstep) + 1
    bR = np.linspace(bmin,bmax,bN, endpoint = False)
  elif cost == "quadratic" :
    assert np.abs(1/astep - round(1/astep) ) < 1e-6, f"astep={astep} should have integer inversion"
    assert np.abs(1/sstep - round(1/sstep) ) < 1e-6, f"sstep={sstep} should have integer inversion"
    
    bstep = c_quad_coef*(astep*sstep)**2  # this assumes quadratic cost
    bR = np.arange(bmin,bmax,bstep)

  K = s0

  aR = np.linspace(0,alim,aN, endpoint= False)
  
    
  assert cliped_reward in set([True,False])
  #cliped_reward = False #False #True

  #epsilon = 0.0 # 0.0 turns off regularization with uniform
  env = Container()
  #env.sN = sN
  #env.sstep = sstep
  env.sR = sR
  env.s0 = s0
  #env.smax = smax
  env.skernel2 = skernel2

  #env.aN = aN
  env.alim = alim
  #env.astep = astep
  env.T = T
  env.bR = bR
  env.aR = aR
  env.K = K

  env.isbounded = isbounded

  # prarameters of various cost functions are stored in env object (used for setting money discretization grid)
  env.fixed_cost = fixed_cost
  env.c_treshold = c_treshold
  env.c_0 = c_0
  env.c_1 = c_1
  env.c_1s = c_1s
  env.c_quad_coef = c_quad_coef

  # prarameters of various utility functions
  env.exp_ulility_alpha = 0.5 #1.0

  env.cost = cost  
  env.cliped_reward = cliped_reward
  env.epsilon = epsilon

  env.utility_name = utility_name
  assert env.utility_name in set(["quadratic","exponential"])

  if env.utility_name == "quadratic" :
    if env.cliped_reward :
      env.utility = quadratic_utility_clipped.__get__(env) # here we bind the method to instance env, because clipped version needs acess to env atributes
    else :  
      env.utility = quadratic_utility.__get__(env)
  elif env.utility_name == "exponential" :
    assert not env.cliped_reward
    env.utility = exp_utility.__get__(env)


  if cost == "fixed" :
    env.cost_function = cost_fixed.__get__(env) # storing cost parameters inside env object
  elif cost == "2slopes":
    env.cost_function = cost_2slopes.__get__(env)
  elif cost == "2slopes10":
    env.cost_function = cost_2slopes10.__get__(env)
  elif cost == "2slopes50":
    env.cost_function = cost_2slopes50.__get__(env)
  elif cost == "quadratic":
    env.cost_function = cost_quadratic.__get__(env)
  

  # deault behaviour for dynamic.prog./simulation
  # by default the computation is assumed precise, ie., it allways snap to the grid

  # non-uniformity error
  env.rise_discretization_error = True

  # value is out of grid edges (this is just for simulation and unbounded setting now)
  env.rise_out_of_range_error = True

  # env recomended initial state (indices) --- just for plotting and simulation
  env.t0,env.si0,env.ai0,env.bi0 = 0,4,8,900*2 # current (#sent to chat)

  return env

  
def make_env_nonconvex_cost():
  env = make_env_factory("quadratic", sN = 9, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 20, alim = 1, T = 5, isbounded = False, cost = "2slopes", cliped_reward = False, epsilon = 0.0)
  return env

def make_env_nonconvex_cost_clipped_reward():
  env = make_env_factory("quadratic", sN = 9, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 20, alim = 1, T = 5, isbounded = False, cost = "2slopes", cliped_reward = True, epsilon = 0.0)
  return env

def make_env_nonconvex_cost_trinomial(clipped_reward, plot_type_suboption):  
  sstep = 1
  p_up = 0.2
  p_down = 0.2
  if plot_type_suboption in set(["quadratic_utility", "quadratic_utility_gbmapprox"]) :
    env = make_env_factory("quadratic", sN = 9, sstep = sstep, p_up = p_up, p_down = p_down, aN = 20, alim = 1, T = 5, isbounded = False, cost = "2slopes", cliped_reward = clipped_reward, epsilon = 0.0)
  elif plot_type_suboption == "increasing_utility" :
    env = make_env_factory("exponential", sN = 9, sstep = sstep, p_up = p_up, p_down = p_down, aN = 20, alim = 1, T = 5, isbounded = False, cost = "2slopes", cliped_reward = clipped_reward, epsilon = 0.0)


  sN = 21 #21 #9 #21
 
  ls0 = np.log(env.s0)
  lsstep = np.log( (env.s0+sstep)/env.s0 )

  sR = np.exp( ( np.linspace(0,sN-1,sN)-sN//2 )*lsstep + ls0 )
  # s0, K is the same up to numerical error


  # trinomial model
  skernel2 = np.zeros([sN,sN])
  p_mid = 1-p_up-p_down
  skernel2 += np.eye(sN,sN,0)*p_mid
  skernel2 += np.eye(sN,sN,-1)*p_up
  skernel2 += np.eye(sN,sN,1)*p_down
  skernel2[0,0] += p_down
  skernel2[sN-1,sN-1] += p_up


  # bN = 2*len(env.bR)
  # bmax = -2*env.bR[0]
  # env.bR = np.linspace(-bmax,bmax,bN, endpoint = False)

  env.sR = sR
  env.skernel2 = skernel2

  # allow for roundig to grid
  env.rise_discretization_error = False

  # still shout when simulation comes out of grid edges
  env.rise_out_of_range_error = True

  env.si0 = sN//2
  #env.bi0 = bN//2
  #env.T = 2

  return env



def make_env_constraint( plot_type_suboption ):
  
  #env = make_env_factory(sN = 2, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 20, alim = 2, T = 5, isbounded = True, cost = "quadratic", cliped_reward = False, epsilon = 0.0, bmax=4,bmin=0)
  #env = make_env_factory(sN = 2, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 20, alim = 2, T = 5, isbounded = True, cost = "2slopes", cliped_reward = False, epsilon = 0.0, bmax=4,bmin=0)

  if plot_type_suboption == "bottom":
    # bottom money constraint + nonconvex cost
    env = make_env_factory("exponential", sN = 2, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 40, alim = 2, T = 5, isbounded = True, cost = "2slopes10", cliped_reward = False, epsilon = 0.0, bmax=4,bmin=0)
  elif plot_type_suboption == "top":
    env = make_env_factory("exponential", sN = 2, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 40, alim = 2, T = 5, isbounded = True, cost = "quadratic", cliped_reward = False, epsilon = 0.0, bmax=4,bmin=0)

  # smax = np.max(env.sR)  
  # bmax = 4 #2*smax*env.alim
  # bmin = 0

  # astep = env.aR[1]-env.aR[0]
  # sstep = env.sR[1]-env.sR[0]
  
  # # all steps should have integer inv
  # assert np.abs(1/astep - round(1/astep) ) < 1e-6, f"astep={astep} should have integer inversion"
  # assert np.abs(1/sstep - round(1/sstep) ) < 1e-6, f"sstep={sstep} should have integer inversion"
  
  # bstep = env.c_quad_coef*(astep*sstep)**2  # this assumes quadratic cost

  # bR = np.arange(bmin,bmax,bstep)
  # env.bR = bR

  #env.utility = log_utility.__get__(env)
  #env.utility = exp_utility.__get__(env) #log_utility.__get__(env)

  return env


def make_env_non_inc_util_non_convex_cost( plot_type_suboption ):
  

  if plot_type_suboption == "noninc_util":
    # bottom money constraint + nonconvex cost
    env = make_env_factory("quadratic", sN = 2, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 20, alim = 1, T = 5, isbounded = False, cost = "quadratic", cliped_reward = False, epsilon = 0.0)    
  elif plot_type_suboption == "non_convex_cost":
    env = make_env_factory("exponential", sN = 2, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 20, alim = 1, T = 5, isbounded = False, cost = "2slopes", cliped_reward = False, epsilon = 0.0)    
  elif plot_type_suboption == "no_problem":
    env = make_env_factory("exponential", sN = 2, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 20, alim = 1, T = 5, isbounded = False, cost = "quadratic", cliped_reward = False, epsilon = 0.0)


  #make_env_nonconvex_cost():
  #env = make_env_factory("quadratic", sN = 9, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 20, alim = 1, T = 5, isbounded = False, cost = "2slopes", cliped_reward = False, epsilon = 0.0)


  return env



########################

def make_env_constraint_realistic( plot_type_suboption ):
  

  if plot_type_suboption == "bottom":
    # bottom money constraint + nonconvex cost
    #env = make_env_factory("exponential", sN = 2, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 40, alim = 2, T = 5, isbounded = True, cost = "2slopes10", cliped_reward = False, epsilon = 0.0, bmax=4,bmin=0)
    env = make_env_factory("exponential", sN = 9, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 40, alim = 2, T = 5, isbounded = True, cost = "2slopes50", cliped_reward = False, epsilon = 0.0, bmax=8,bmin=0)
  elif plot_type_suboption == "top":
    #env = make_env_factory("exponential", sN = 2, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 40, alim = 2, T = 5, isbounded = True, cost = "quadratic", cliped_reward = False, epsilon = 0.0, bmax=4,bmin=0)
    env = make_env_factory("exponential", sN = 9, sstep = 1, p_up = 0.2, p_down = 0.2, aN = 40, alim = 2, T = 5, isbounded = True, cost = "quadratic", cliped_reward = False, epsilon = 0.0, bmax=4,bmin=0)

  #env = make_env_factory("exponential", sN = 9, sstep = sstep, p_up = p_up, p_down = p_down, aN = 20, alim = 1, T = 5, isbounded = False, cost = "2slopes", cliped_reward = clipped_reward, epsilon = 0.0)

  sstep = 1
  p_up = 0.2
  p_down = 0.2

  sN = 21 #21 #9 #21
 
  ls0 = np.log(env.s0)
  lsstep = np.log( (env.s0+sstep)/env.s0 )

  sR = np.exp( ( np.linspace(0,sN-1,sN)-sN//2 )*lsstep + ls0 )
  # s0, K is the same up to numerical error


  # trinomial model
  skernel2 = np.zeros([sN,sN])
  p_mid = 1-p_up-p_down
  skernel2 += np.eye(sN,sN,0)*p_mid
  skernel2 += np.eye(sN,sN,-1)*p_up
  skernel2 += np.eye(sN,sN,1)*p_down
  skernel2[0,0] += p_down
  skernel2[sN-1,sN-1] += p_up


  # bN = 2*len(env.bR)
  # bmax = -2*env.bR[0]
  # env.bR = np.linspace(-bmax,bmax,bN, endpoint = False)

  env.sR = sR
  env.skernel2 = skernel2

  # allow for roundig to grid
  env.rise_discretization_error = False

  # still shout when simulation comes out of grid edges
  env.rise_out_of_range_error = True

  env.si0 = sN//2
  #env.bi0 = bN//2
  #env.T = 2


  return env
