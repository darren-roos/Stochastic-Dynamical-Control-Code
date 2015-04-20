# Particle Swarm Optimisation
module PSO

using PF
using Distributions
using Reactor
using NLsolve

type Particle
  pos
  vel
  prevbest
  prevbestpos
end

type Solution
  pos
  posfit
end


function fitness(particles, us, ysp, usp, Q, R, plantdist, model)
  # This should ideally be run a few times to get the MC expected value.
  # NOTE: we use the standard quadratic function in the expectation i.e.
  # E[(x-y)'Q(x-y)]

  parts = copy(particles.x)
  fit = 0.0
  N = length(us)
  nX, nP = size(parts)

  for p=1:nP # initial error
    fit = fit + particles.w[p]*((parts[:, p]-ysp)'*Q*(parts[:, p]-ysp))
  end

  for n=1:N # predicted error
    PF.predict!(parts, us[n], plantdist, model) # modifies parts
    for p=1:nP
      fit = fit + particles.w[p]*((parts[:, p]-ysp)'*Q*(parts[:, p]-ysp)) + (us[n]-usp)'*R*(us[n]-usp)
    end
  end

  return fit/N
end

function offset(y_concentration, model)
  # Finds the steady state given the controller input. Assume only concentration may be controlled.
  f!(x, fvec) = Reactor.reactor_func!([y_concentration, x[1]], x[2], model, fvec)
  offset_results = nlsolve(f!, [0.5, 0.0])
  return offset_results
end

function createswarm(nP, nX, umin, umax)
  # Initialise the swarm with random positions and velocities within a range.
  swarm = Array(Particle, nP) # create swarm
  for p=1:nP
    pnow = randomr(nX, umin, umax)
    swarm[p] = Particle(pnow, randomr(nX, umin, umax), 0.0, pnow)
  end
  sol = Solution(randomr(nX, umin, umax), 0.0)
  return swarm, sol
end

function initswarm(nP, nX, umin, umax, particles, ysp, usp, Q, R, plantdist, model)

  swarm = Array(Particle, nP) # create swarm
  sol = Solution(randomr(nX, umin, umax), Inf)

  for p=1:nP
    pnow = randomr(nX, umin, umax)
    swarm[p] = Particle(pnow, 0.5.*randomr(nX, umin, umax), 0.0, pnow)
    swarm[p].prevbest = fitness(particles, swarm[p].pos, ysp, usp, Q, R, plantdist, model)
    if swarm[p].prevbest[1] < sol.posfit[1]
      sol.pos = swarm[p].prevbestpos
      sol.posfit = swarm[p].prevbest
    end
  end
  return swarm, sol
end

function updateswarm!(swarm, sol, particles, ysp, usp, Q, R, plantdist, model)
# Using optimum parameters
  nP = length(swarm)
  nX = length(swarm[1].pos)
  for p=1:nP
    swarm[p].vel = 0.7298.*swarm[p].vel + randomr(nX, 0.0, 1.49618).*(swarm[p].prevbestpos - swarm[p].pos) +  randomr(nX, 0.0, 1.49618).*(sol.pos - swarm[p].pos)
    swarm[p].pos = swarm[p].pos + swarm[p].vel

    tempfitness = fitness(particles, swarm[p].pos, ysp, usp, Q, R, plantdist, model)

    if tempfitness[1] < sol.posfit[1]
      sol.pos = swarm[p].pos
      sol.posfit = tempfitness
    end

    if tempfitness[1] < swarm[p].prevbest[1]
      swarm[p].prevbestpos = swarm[p].pos
      swarm[p].prevbest = tempfitness
    end

  end

end

function optimise!(swarm, sol, particles, ysp, usp, Q, R, plantdist,model)
  # Run the PSO
  for k=1:50
    updateswarm!(swarm, sol, particles, ysp, usp, Q, R, plantdist, model)
    sol.pos[1]
  end
  return sol.pos[1]
end

function randomr(nX, xmin, xmax)
  # Returns a uniformly distributed random number in the range (xmin, xmax]
  return rand(nX).*(xmax-xmin) .+ xmin
end


end # PSO module
