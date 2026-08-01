
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import envs


#np.random.seed(1234)


show_figures = True

ussage_message = """
Usage:

python main.py plot_type plot_type_suboption [donotshowfigs]

where "plot_type plot_type_suboption" is one of:
nonconvex_cost quadratic_utility
constraint bottom
constraint top
non_inc_util_non_convex_cost noninc_util
non_inc_util_non_convex_cost non_convex_cost
constraint_realistic bottom

The optional donotshowfigs switch disibles plotting figures on screen in blocking mode (useful for noninteractive runs).
"""

assert len(sys.argv) in set([3,4]), "The script accepts two or three arguments, but different number of arguments were given. Abborting."+ussage_message
assert sys.argv[1]+" "+sys.argv[2] in set(["nonconvex_cost quadratic_utility",
"constraint bottom", "constraint top",
"non_inc_util_non_convex_cost noninc_util", "non_inc_util_non_convex_cost non_convex_cost",
"constraint_realistic bottom"]), f"The {sys.argv[1]} {sys.argv[2]} are not a valid arguments. Aborting."+ussage_message
if len(sys.argv) == 4:
  assert sys.argv[3] == "donotshowfigs", f"The optinal third argument could be only donotshowfigs but {sys.argv[3]} was given. Aborting."+ussage_message
  show_figures = False
plot_type = sys.argv[1]
plot_type_suboption = sys.argv[2]



try:
  os.mkdir("fig")
except FileExistsError as error: #OSError
  #print(error)
  pass


plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({'font.size': 7}) # was 17 default 10; this is for default 0.5\textwidth scale
plt.rcParams.update({'axes.labelsize': 10})
plt.rcParams.update({'legend.fontsize': 10})
plt.rcParams.update({'xtick.labelsize': 8})
plt.rcParams.update({'ytick.labelsize': 8})
plt.rcParams.update({'xtick.major.size':1.5})   #3.5 default
plt.rcParams.update({'ytick.major.size':1.5})
plt.rcParams.update({'xtick.minor.size':0.9})   #2 default
plt.rcParams.update({'ytick.minor.size':0.9})
plt.rcParams.update({'lines.linewidth':0.75})    #1.5 default
plt.rcParams.update({'axes.linewidth':0.25})    #0.8 default
plt.rcParams.update({'xtick.major.width':0.25}) #0.8 default
plt.rcParams.update({'ytick.major.width':0.25})
plt.rcParams.update({'xtick.minor.width':0.20}) #0.6 default
plt.rcParams.update({'ytick.minor.width':0.20})

plt.rcParams.update({'lines.markeredgewidth':0.35}) #1.0 default
plt.rcParams.update({'lines.markersize':3})         #6 default
plt.rcParams.update({'legend.fontsize':10})  # default




def get_values(env) :

  sR,aR,bR = env.sR,env.aR,env.bR
  sN,aN,bN = len(env.sR),len(aR),len(bR)
  T = env.T
  bmin = np.min(bR)
  bmax = np.max(bR)
  bstep = bR[1]-bR[0]
  skernel2 = env.skernel2

  Q = np.zeros([T,sN,aN,bN,aN])

  si,ai,bi,ai2 = np.mgrid[0:sN,0:aN,0:bN,0:aN]
  s = sR[si]
  n = aR[ai]
  b = bR[bi]
  n_ = aR[ai2]
  for t in range(T-1,-1,-1) :
    print(f"t = {t}")

    c = env.cost_function(np.abs(n_ - n), s)
    b_ = b - (n_-n)*s - c
    if env.rise_discretization_error :
      assert np.sum(np.abs( np.round((b_ - bmin) / bstep) -  (b_ - bmin) / bstep ) >= 1e-6) == 0  , f"Discretization error" # t,si,ai,bi,ai2={(t,si,ai,bi,ai2)}"
    bi2 = (np.round((b_ - bmin) / bstep)).astype(np.int64)
    mask = (1 - (bi2 > (bN-1))) *  (1-  (bi2 < 0)) > 0
    if env.isbounded :
      Q[t][np.logical_not(mask)] = -np.inf
    else :
      Q[t][np.logical_not(mask)] = np.nan

    if t == T-1 :  
      for si2 in range(sN) :
        s_ = sR[si2]      
        mask_prob0 = skernel2[si2][si] != 0
        qinc = np.zeros(si.shape)
        qinc[mask_prob0] = skernel2[si2][si][mask_prob0]*env.utility( n_*s_ + b_ - np.max([s_-env.K,0]) )[mask_prob0]
        if env.isbounded :
          Q[t][mask] += qinc[mask]
        else :
          Q[t] += qinc
    else :
      for si2 in range(sN) :
        s_ = sR[si2]        
        Qinf2zero = Q[t+1,si2].copy()
        Qinf2zero[Qinf2zero==-np.inf] = 0
        Vt1si2 = (1-env.epsilon)*Q[t+1,si2].max(axis = -1) + (env.epsilon/aN)*Qinf2zero.sum(axis = -1)  # + (env.epsilon/aN)*Q[t+1,si2].sum(axis = -1)        
        V = Vt1si2[ai2[mask],bi2[mask]]
        assert not ( (np.sum(V == -np.inf) > 0 )  ), "Invalid value function V" 
        mask_prob0 = np.logical_and( skernel2[si2][si] != 0, mask )
        Q[t][mask_prob0] += skernel2[si2][si[mask_prob0]]*Vt1si2[ai2[mask_prob0],bi2[mask_prob0]]

  # opt V and opt pi
  V = np.max(Q, axis = -1)
  Ve = np.max(Q, axis = -1, keepdims = True )
  mask = np.logical_not(np.isnan(V))
  pi = np.zeros([T,sN,aN,bN,aN])
  pi[...] = np.nan
  pi[mask] = Q[mask] == Ve[mask]
  pi[mask] = pi[mask]/pi[mask].sum(axis = -1, keepdims = True )

  return Q,V,pi



def local_max_detection(Q) :
  #local max detection  
  p0 = Q[...,0:-2]
  p1 = Q[...,1:-1]
  p2 = Q[...,2:]  
  mask = np.logical_not ( np.isnan(p0) | np.isnan(p1) | np.isnan(p2) | (p0 == -np.inf) | (p1 == -np.inf) | (p2 == -np.inf) )
  locmask = np.zeros(p0.shape, dtype = np.bool_)
  locmask[mask] = (p1[mask] - p0[mask] > 0) & (p2[mask] - p1[mask] < 0)
  lmcnt = locmask.sum(axis = -1).astype(np.int64)
  return lmcnt




#plot_type = "nonconvex_cost"
#plot_type = "constraint"
#plot_type = "non_inc_util_non_convex_cost"
#plot_type = "constraint_realistic"


assert plot_type in set(["nonconvex_cost", "constraint", "non_inc_util_non_convex_cost", "constraint_realistic" ])
print(f"plot_type={plot_type}")



if plot_type == "nonconvex_cost" :

  #plot_type_suboption = "quadratic_utility"
  #plot_type_suboption = "increasing_utility"
  print(f"plot_type_suboption={plot_type_suboption}")


  assert plot_type_suboption in set(["quadratic_utility","increasing_utility"])

 
  if plot_type_suboption in set(["quadratic_utility"]) :
    #env = envs.make_env_nonconvex_cost()               # without Al0 clipping
    env = envs.make_env_nonconvex_cost_clipped_reward() # with Al0 clipping
    #env = envs.make_env_nonconvex_cost_trinomial(clipped_reward = True, plot_type_suboption = plot_type_suboption )
  elif plot_type_suboption == "increasing_utility" :
    env = envs.make_env_nonconvex_cost_trinomial(clipped_reward = False, plot_type_suboption = plot_type_suboption )


  Q,V,pi = get_values(env)
  lmcnt = local_max_detection(Q)

  print(f"nan concentration by t:")              
  for t in range(env.T) :
    print(f"t={t}, nanc={(np.isnan(Q[t]).sum()/len(Q[t].flatten()))}",end="")
  print("\n")

  if plot_type_suboption == "quadratic_utility" :
    t0,si0,ai0,bi0 = env.t0,env.si0,env.ai0,env.bi0
  elif plot_type_suboption == "increasing_utility" :
    t0,si0,ai0,bi0 = env.t0,env.si0,env.ai0,env.bi0
    ai0 = ai0 - 3

  plt.rcParams.update({'font.size': 12}) # default 10

  plt.figure()
  plt.plot(env.aR,Q[t0,si0,ai0,bi0].T)
  # plt.xlabel(r'$a_0$') 
  # plt.ylabel(r'$Q^*(s_0,a_0)$')
  plt.xlabel(r'$a_0$') 
  plt.ylabel(r'$Q^*(s_0,a_0)$')
  plt.savefig(f"fig/quadutil_noncovexcost.pdf")
  if show_figures: 
    plt.show()



if plot_type == "constraint" :
  #plot_type_suboption = "bottom"
  #plot_type_suboption = "top"
  print(f"plot_type_suboption={plot_type_suboption}")

  assert plot_type_suboption in set(["top","bottom"])

  env = envs.make_env_constraint( plot_type_suboption ) # nonconve cost for bottom

  Q,V,pi = get_values(env)
  lmcnt = local_max_detection(Q)
  
  ai = 2*15

  h = 1
  if plot_type_suboption == "bottom" :
    si = 0
    bi = 14
    levels = 10
  elif plot_type_suboption == "top"  :
    si = 1
    bi = int(np.round(len(env.bR)*0.95))
    levels = 40

  inches_per_pt = 1.0/72.27
  linewidth = inches_per_pt*433.62
  two_figs_in_row = 0.48*linewidth # originaly setting: 8,8
  two_figs_in_row_axis = (0.190,0.130,0.95-0.190,0.95-0.130)
  two_figs_in_row_axis_spec = (0.190+0.040,0.130,0.95-0.190-0.040,0.95-0.130)
  two_figs_in_row_axis_spec2 = (0.150,0.130,0.95-0.150-0.010,0.95-0.130)



  fig = plt.figure()
  fig.set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_spec2)
  ax = fig.get_axes()[0]


  #T-1 Q
  pcmesh = ax.pcolormesh(env.aR, env.bR, Q[env.T-h,si,ai,:,:])
  ax.contour(env.aR, env.bR, Q[env.T-h,si,ai,:,:], levels = levels, colors = "gray")
  ax.plot([env.aR[0],env.aR[-1]],env.bR[bi]*np.ones([2]),"k-")
  ax.set_xlim(env.aR[0],2)
  fig.colorbar(pcmesh)
  #ax.set_xlabel(r'$a_{n-1}\;(=d_n^1)$')
  ax.set_xlabel(r'$a_{n-1}\;(=d_n)$')
  #ax.set_ylabel(r'$d_{n-1}^0$')
  ax.set_ylabel(r'$b_{n-1}$')
  if plot_type_suboption == "top" :
    plt.savefig(f"fig/constrheattop.pdf")
  if plot_type_suboption == "bottom" : 
    plt.savefig(f"fig/constrheatbottom.pdf")

  # Q plots
  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_spec)
  plt.plot(env.aR, Q[env.T-1,si,ai,bi,:])
  plt.xlim(env.aR[0],2)
  # plt.xlabel(r'$a_{n-1}\;(=d_n^1)$')
  # plt.ylabel(r'$Q^{*}_{n-1}(s_{n-1},a_{n-1})$')
  plt.xlabel(r'$a_{n-1}\;(=d_n)$')
  plt.ylabel(r'$Q^{*}_{n-1}(s_{n-1},a_{n-1})$')

  if plot_type_suboption == "top" :
    plt.savefig(f"fig/constrcuttop.pdf")
  if plot_type_suboption == "bottom" : 
    plt.savefig(f"fig/constrcutbottom.pdf")

  if show_figures: 
    plt.show()






if plot_type == "non_inc_util_non_convex_cost" :


  #plot_type_suboption = "no_problem"
  #plot_type_suboption = "noninc_util"
  #plot_type_suboption = "non_convex_cost"
  print(f"plot_type_suboption={plot_type_suboption}")

  assert plot_type_suboption in set(["noninc_util","non_convex_cost","no_problem"])

  env = envs.make_env_non_inc_util_non_convex_cost( plot_type_suboption )


  Q,V,pi = get_values(env)
  lmcnt = local_max_detection(Q)


  print(f"nan concentration by t:")              
  for t in range(env.T) :
    print(f"t={t}, nanc={(np.isnan(Q[t]).sum()/len(Q[t].flatten()))}",end="")
  print("\n")


  if plot_type_suboption == "no_problem" :
    t0,si0,ai0,bi0 = 4,0,8,4000
  if plot_type_suboption == "noninc_util" :    
    t0,si0,ai0,bi0 = 4,1,11,3470+50
  if plot_type_suboption == "non_convex_cost" :
    t0,si0,ai0,bi0 = 4,0,8,400


  inches_per_pt = 1.0/72.27
  linewidth = inches_per_pt*433.62
  two_figs_in_row = 0.48*linewidth 
  two_figs_in_row_axis = (0.190,0.130,0.95-0.190,0.95-0.130)
  two_figs_in_row_axis_spec = (0.190+0.040,0.130,0.95-0.190-0.040,0.95-0.130)


  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_spec)
  plt.plot(env.aR,Q[t0,si0,ai0,bi0].T)
  # plt.xlabel(r'$a_{n-1}\;(=d_{n}^1)$') 
  # plt.ylabel(r'$Q^{*}_{n-1}(s_{n-1},a_{n-1})$')
  plt.xlabel(r'$a_{n-1}\;(=d_{n})$') 
  plt.ylabel(r'$Q^{*}_{n-1}(s_{n-1},a_{n-1})$')
  plt.xlim(0,1)
  if plot_type_suboption == "no_problem" :
    plt.savefig(f"fig/noproblem.pdf")
  if plot_type_suboption == "noninc_util" : 
    plt.savefig(f"fig/nonincutil.pdf")
  if plot_type_suboption == "non_convex_cost" :
    plt.savefig(f"fig/nonconcost.pdf")
  if show_figures:     
    plt.show()





if plot_type == "constraint_realistic" :
  #plot_type_suboption = "bottom"
  #plot_type_suboption = "top"
  print(f"plot_type_suboption={plot_type_suboption}")

  assert plot_type_suboption in set(["top","bottom"])

  env = envs.make_env_constraint_realistic( plot_type_suboption ) # nonconve cost for bottom

  Q,V,pi = get_values(env)
  lmcnt = local_max_detection(Q)
  
  ai = 30 #2*15

  h = 5 #1
  if plot_type_suboption == "bottom" :
    si = 10 #0
    bi = 65 #45 #14
    levels = 10
  elif plot_type_suboption == "top"  :
    si = 1
    bi = int(np.round(len(env.bR)*0.95))
    levels = 40


  inches_per_pt = 1.0/72.27
  linewidth = inches_per_pt*433.62
  two_figs_in_row = 0.48*linewidth 
  two_figs_in_row_y = 3.5/5*two_figs_in_row
  two_figs_in_row_axis = (0.190,0.130,0.95-0.190,0.95-0.130)
  two_figs_in_row_axis_spec =   (0.190+0.040,0.130,0.95-0.190-0.040,0.95-0.130)
  two_figs_in_row_axis_spec2 =  (0.150,0.130    ,0.95-0.150-0.010,0.95-0.130)
  alittle = 0.075
  two_figs_in_row_axis_spec_ =  (0.190+0.040,0.130+alittle    ,0.95-0.190-0.040,0.95-0.130-alittle)
  two_figs_in_row_axis_spec_2 = (0.150      ,0.130+alittle    ,0.95-0.150-0.010,0.95-0.130-alittle)


  fig = plt.figure()
  fig.set_size_inches(two_figs_in_row, two_figs_in_row_y)
  plt.axes(two_figs_in_row_axis_spec_2)
  ax = fig.get_axes()[0]


  #T-1 Q
  pcmesh = ax.pcolormesh(env.aR, env.bR, Q[env.T-h,si,ai,:,:])
  ax.contour(env.aR, env.bR, Q[env.T-h,si,ai,:,:], levels = levels, colors = "gray")
  ax.plot([env.aR[0],env.aR[-1]],env.bR[bi]*np.ones([2]),"k-")
  ax.set_xlim(env.aR[0],2)
  fig.colorbar(pcmesh)
  # ax.set_xlabel(r'$a_{n-5}\;(=d_{n-4}^1)$')
  # ax.set_ylabel(r'$d_{n-5}^0$')
  ax.set_xlabel(r'$a_{n-5}\;(=d_{n-4})$')
  ax.set_ylabel(r'$b_{n-5}$')
  if plot_type_suboption == "top" :
    plt.savefig(f"fig/constrheattop5.pdf")
  if plot_type_suboption == "bottom" : 
    plt.savefig(f"fig/constrheatbottom5.pdf")


  # Q plots
  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row_y)
  plt.axes(two_figs_in_row_axis_spec_)
  plt.plot(env.aR, Q[env.T-1,si,ai,bi,:])
  plt.xlim(env.aR[0],2)
  # plt.xlabel(r'$a_{n-5}\;(=d_{n-4}^1)$')
  # plt.ylabel(r'$Q^{*}_{n-5}(s_{n-5},a_{n-5})$')
  plt.xlabel(r'$a_{n-5}\;(=d_{n-4})$')
  plt.ylabel(r'$Q^{*}_{n-5}(s_{n-5},a_{n-5})$')
  if plot_type_suboption == "top" :
    plt.savefig(f"fig/constrcuttop5.pdf")
  if plot_type_suboption == "bottom" : 
    plt.savefig(f"fig/constrcutbottom5.pdf")

  if show_figures:   
    plt.show()
