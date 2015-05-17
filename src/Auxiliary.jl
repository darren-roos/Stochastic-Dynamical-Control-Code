module Auxiliary

using Distributions
using KernelDensity
using PyPlot

warn("Auxiliary is hardcoded for the CSTR!")
function KL(part_states, part_weights, m, S, temp_states)
  # Discrete Kullback-Leibler divergence test wrt a multivariate normal model.

  sweights = 1.0 - sum(part_weights)
  N = length(part_weights)
  if sweights < 0.0 # add some robustness here (very crude)
    part_weights = part_weights .+ (1.0/N)*sweights
    w = string("Particle weights adjusted by ", sweights, " in Auxiliary!")
    warn(w)
  elseif sweights > 0.0
    part_weights = part_weights .+ (1.0/N)*sweights
    w = string("Particle weights adjusted by ", sweights, " in Auxiliary!")
    warn(w)
  end

  dnorm = MvNormal(m, S)
  dcat = Categorical(part_weights)

  for k=1:N
    i = rand(dcat)
    temp_states[:, k] = part_states[:, i]
  end
  estden = kde(temp_states')

  kldiv = 0.0
  for k=1:N
    #draw from samplesw
    kldiv += -log(pdf(dnorm, temp_states[:, k])) + log(pdf(estden, temp_states[1, k], temp_states[2, k]))
  end

  return (1.0/N)*kldiv
end

function showEstimatedDensity(part_states, part_weights, temp_states)
  # Discrete Kullback-Leibler divergence test wrt a multivariate normal model.

  sweights = 1.0 - sum(part_weights)
  N = length(part_weights)
  if sweights < 0.0 # add some robustness here (very crude)
    part_weights = part_weights .+ (1.0/N)*sweights
    w = string("Particle weights adjusted by ", sweights, " in Auxiliary!")
    warn(w)
  elseif sweights > 0.0
    part_weights = part_weights .+ (1.0/N)*sweights
    w = string("Particle weights adjusted by ", sweights, " in Auxiliary!")
    warn(w)
  end

  dcat = Categorical(part_weights)

  for k=1:N
    i = rand(dcat)
    temp_states[:, k] = part_states[:, i]
  end
  estden = kde(temp_states')
  rc("font", family="serif", size=24)
  figure() # new figure otherwise very cluttered
  contour(estden)
end

end # module