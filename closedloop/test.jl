# Controller using the linear reactor model measuring both concentration and temperature.

include("../params.jl") # load all the parameters and modules

using JuMP
using Mosek


# Get the linear model
linsystems = Reactor.getNominalLinearSystems(h, cstr_model) # cstr_model comes from params.jl
opoint = 2 # the specific operating point we are going to use for control

init_state = [0.5, 450] # random initial point near operating point

# Set the state space model
A = linsystems[opoint].A
B = linsystems[opoint].B
b = linsystems[opoint].b # offset from the origin

# Set point
# ysp = linsystems[1].op[1] - b[1] # Low concentration
ysp = linsystems[2].op[1] - b[1] # Medium concentration
# ysp = linsystems[3].op[1] - b[1] # High concentration
# ysp = 0.1 - b[1]

# Create the controller
H = [1.0 0.0] # only attempt to control the concentration
x_off, u_off = LQR.offset(A,B,C2,H, ysp) # control offset
K = LQR.lqr(A, B, QQ, RR) # controller

# Set up the KF
kf_cstr = LLDS.llds(A, B, C2, Q, R2) # set up the KF object (measuring both states)
state_noise_dist = MvNormal(Q)
meas_noise_dist = MvNormal(R2)

# First time step of the simulation
xs[:,1] = init_state # set simulation starting point to the random initial state
ys2[:, 1] = C2*xs[:, 1] + rand(meas_noise_dist) # measure from actual plant
kfmeans[:, 1], kfcovars[:,:, 1] = LLDS.init_filter(init_state-b, init_state_covar, ys2[:, 1]-b, kf_cstr) # filter

# Setup MPC
horizon = 300

m = Model(solver=MosekSolver(LOG=0)) # chooses optimiser by itself

@defVar(m, x[1:2, 1:horizon])
@defVar(m, u[1:horizon-1])

# add dynamics constraints

# k = 2
@addConstraint(m, x[1, 1] == kfmeans[1, 1])
@addConstraint(m, x[2, 1] == kfmeans[2, 1])
@addConstraint(m, x[1, 2] == A[1,1]*kfmeans[1, 1] + A[1,2]*kfmeans[2, 1]+ B[1]*u[1])
@addConstraint(m, x[2, 2] == A[2,1]*kfmeans[1, 1] + A[2,2]*kfmeans[2, 1] + B[2]*u[1])

for k=3:prediction_horizon
  @addConstraint(m, x[1, k] == A[1,1]*x[1, k-1] + A[1,2]*x[2, k-1] + B[1]*u[k-1])
  @addConstraint(m, x[2, k] == A[2,1]*x[1, k-1] + A[2,2]*x[2, k-1] + B[2]*u[k-1])
end

@setObjective(m, Min, sum{x[1, i]*QQ[1]*x[1, i] + u[i]*RR*u[i], i=1:horizon-1} + x[1, horizon]*QQ[1]*x[1, horizon])

status = solve(m)

us[1] = getValue(u[1]) # get the controller input

for t=2:N
  xs[:, t] = Reactor.run_reactor(xs[:, t-1], us[t-1], h, cstr_model) + rand(state_noise_dist) # actual plant
  ys2[:, t] = C2*xs[:, t] + rand(meas_noise_dist) # measure from actual plant
  kfmeans[:, t], kfcovars[:,:, t] = LLDS.step_filter(kfmeans[:, t-1], kfcovars[:,:, t-1], us[t-1], ys2[:, t]-b, kf_cstr)

  # Compute controller action
  m = Model(solver=MosekSolver(LOG=0)) # chooses optimiser by itself
  @defVar(m, x[1:2, 1:horizon])
  @defVar(m, u[1:horizon-1])
  @addConstraint(m, x[1, 1] == kfmeans[1, t])
  @addConstraint(m, x[2, 1] == kfmeans[2, t])
  @addConstraint(m, x[1, 2] == A[1,1]*kfmeans[1, t] + A[1,2]*kfmeans[2, t]+ B[1]*u[1])
  @addConstraint(m, x[2, 2] == A[2,1]*kfmeans[1, t] + A[2,2]*kfmeans[2, t] + B[2]*u[1])
  for k=3:prediction_horizon
    @addConstraint(m, x[1, k] == A[1,1]*x[1, k-1] + A[1,2]*x[2, k-1] + B[1]*u[k-1])
    @addConstraint(m, x[2, k] == A[2,1]*x[1, k-1] + A[2,2]*x[2, k-1] + B[2]*u[k-1])
  end
  @setObjective(m, Min, sum{x[1, i]*QQ[1]*x[1, i] + u[i]*RR*u[i], i=1:horizon-1} + x[1, horizon]*QQ[1]*x[1, horizon])
  status = solve(m)
  us[t] = getValue(u[1]) # get the controller input

end
kfmeans = kfmeans .+ b

# # Plot the results
Results.plotTracking(ts, xs, ys2, kfmeans, us, 2)
