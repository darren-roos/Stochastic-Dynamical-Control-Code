# Controller using the linear reactor model measuring both concentration and temperature.

tend = 200
include("closedloop_params.jl") # load all the parameters and modules

init_state = [0.55, 450] # random initial point near operating point

# Initialise the PF
nP = 200 # number of particles.
prior_dist = MvNormal(init_state, init_state_covar) # prior distribution
particles = PF.init_PF(prior_dist, nP, 2) # initialise the particles
state_noise_dist = MvNormal(Q) # state distribution
meas_noise_dist = MvNormal(R2) # measurement distribution

f(x, u, w) = Reactor.run_reactor(x, u, h, cstr_model) + w
g(x) = C2*x # state observation

cstr_pf = PF.Model(f,g) # PF object

# Get the linear model
linsystems = Reactor.getNominalLinearSystems(h, cstr_model) # cstr_model comes from params.jl
opoint = 2 # the specific operating point we are going to use for control
# Set the state space model
A = linsystems[opoint].A
B = linsystems[opoint].B
b = linsystems[opoint].b # offset from the origin

# Set point
ysp = linsystems[2].op[1] - b[1] # Medium concentration

# Create the controller
H = [1.0 0.0] # only attempt to control the concentration
x_off, u_off = LQR.offset(A,B,C2,H, ysp) # control offset
K = LQR.lqr(A, B, QQ, RR) # controller


# First time step of the simulation
xs[:,1] = init_state # set simulation starting point to the random initial state
ys2[:, 1] = C2*xs[:, 1] + rand(meas_noise_dist) # measure from actual plant
PF.init_filter!(particles, 0.0, ys2[:, 1], meas_noise_dist, cstr_pf)
pfmeans[:,1], pfcovars[:,:,1] = PF.getStats(particles)

us[1] = -K*(pfmeans[:, 1] - b - x_off) + u_off # controller action

for t=2:N
  xs[:, t] = Reactor.run_reactor(xs[:, t-1], us[t-1], h, cstr_model) + rand(state_noise_dist) # actual plant

  ys2[:, t] = C2*xs[:, t] + rand(meas_noise_dist) # measure from actual plant
  PF.filter!(particles, us[t-1], ys2[:, t], state_noise_dist, meas_noise_dist, cstr_pf)
  pfmeans[:,t], pfcovars[:,:,t] = PF.getStats(particles)

  # Compute controller action
  if t%10==0
    us[t] = -K*(pfmeans[:, t] - b - x_off) + u_off # controller action
  else
    us[t] = us[t-1]
  end
end

# Plot the results
Results.plotTracking(ts, xs, ys2, pfmeans, us, 2, ysp+b[1])
Results.calcError(xs, ysp+b[1])
Results.calcEnergy(us, 0.0, h)
