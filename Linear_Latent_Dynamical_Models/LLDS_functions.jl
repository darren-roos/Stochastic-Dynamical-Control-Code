module LLDS_functions

using Distributions

immutable LLDS{T}
  # Assume zero mean transition and emission functions.
  # The linear latent dynamical system should have the
  # state space form:
  # x(t+1) = A*x(t) + Bu(t) + b + Q
  # y(t+1) = C*x(t+1) + R (there could be Du(t) term here but we assume the inputs don't affect
  # the measurements directly.)
  # I assume the simplest model I will deal with has matrix A, B and float C therefore
  # the slightly parametric type. Also that specific simple case has only one input.
  # this is to avoid ugly notation later.
  A :: Array{Float64, 2}
  B :: Array{Float64, 2}
  b :: Array{Float64, 1}
  C :: Array{Float64, 2}
  Q :: Array{Float64, 2} # Process Noise
  R :: T # Measurement Noise VARIANCE
end

function step(xprev::Array{Float64,1}, uprev, model::LLDS{Array{Float64, 2}})
  # Controlled, move multivariate model one time step forward.

  xnow = model.A*xprev + model.B*uprev + model.b
  ynow = model.C*xnow

  return xnow,  ynow
end

function step(xprev::Array{Float64,1}, uprev, model::LLDS{Float64})
  # Controlled, move multivariate model one time step forward.

  xnow = model.A*xprev + model.B*uprev + model.b
  ynow = model.C*xnow

  return xnow,  ynow
end


function init_filter(initmean::Array{Float64, 1}, initvar::Array{Float64, 2}, ynow::Array{Float64, 1}, model::LLDS{Array{Float64, 2}})
  # Initialise the filter. No prediction step, only a measurement update step.
  updatedMean ::Array{Float64, 1}, updatedVar :: Array{Float64, 2} = step_update(initmean, initvar, ynow, model)
  return updatedMean, updatedVar
end

function init_filter(initmean::Array{Float64, 1}, initvar::Array{Float64, 2}, ynow::Float64, model::LLDS{Float64})
  # Initialise the filter. No prediction step, only a measurement update step.
  updatedMean ::Array{Float64, 1}, updatedVar :: Array{Float64, 2} = step_update(initmean, initvar, ynow, model)
  return updatedMean, updatedVar
end

function step_filter(prevmean::Array{Float64, 1}, prevvar::Array{Float64, 2}, uprev::Array{Float64,1}, ynow::Array{Float64, 1}, model::LLDS{Array{Float64, 2}})
  # Return the posterior over the current state given the observation and previous
  # filter result.
  pmean :: Array{Float64, 1}, pvar :: Array{Float64, 2} = step_predict(prevmean, prevvar, uprev, model)
  updatedMean ::Array{Float64, 1}, updatedVar :: Array{Float64, 2} = step_update(pmean, pvar, ynow, model)
  return updatedMean, updatedVar
end

function step_filter(prevmean::Array{Float64, 1}, prevvar::Array{Float64, 2}, uprev::Float64, ynow::Float64, model::LLDS{Float64})
  # Return the posterior over the current state given the observation and previous
  # filter result.

  pmean :: Array{Float64, 1}, pvar :: Array{Float64, 2} = step_predict(prevmean, prevvar, uprev, model)
  updatedMean ::Array{Float64, 1}, updatedVar :: Array{Float64, 2} = step_update(pmean, pvar, ynow, model)
  return updatedMean, updatedVar
end

function step_predict(xprev::Array{Float64,1}, varprev::Array{Float64, 2}, uprev::Array{Float64,1}, model::LLDS{Array{Float64, 2}})
  # Return the one step ahead predicted mean and covariance.
  pmean = model.A*xprev + model.B*uprev + model.b
  pvar =  model.Q + model.A*varprev*transpose(model.A)
  return pmean, pvar
end

function step_predict(xprev::Array{Float64,1}, varprev::Array{Float64, 2}, uprev::Float64, model::LLDS{Float64})
  # Return the one step ahead predicted mean and covariance.
  pmean = model.A*xprev + model.B[:, 1]*uprev + model.b # fix so that Array{Float64, 1} output
  pvar =  model.Q + model.A*varprev*transpose(model.A)
  return pmean, pvar
end

function step_update(pmean::Array{Float64,1}, pvar::Array{Float64, 2}, ymeas::Array{Float64,1}, model::LLDS{Array{Float64, 2}})
  # Return the one step ahead measurement updated mean and covar.
  kalmanGain = pvar*transpose(model.C)*inv(model.C*pvar*transpose(model.C) + model.R)
  ypred = model.C*pmean #predicted measurement
  updatedMean = pmean + kalmanGain*(ymeas - ypred)
  rows, cols = size(pvar)
  updatedVar = (eye(rows) - kalmanGain*model.C)*pvar
  return updatedMean, updatedVar
end

function step_update(pmean::Array{Float64,1}, pvar::Array{Float64, 2}, ymeas::Float64, model::LLDS{Float64})
  # Return the one step ahead measurement updated mean and covar.
  kalmanGain = pvar*transpose(model.C)*inv(model.C*pvar*transpose(model.C) + model.R)
  ypred = model.C*pmean #predicted measurement
  updatedMean = pmean + kalmanGain*(ymeas - ypred)
  rows, cols = size(pvar)
  updatedVar = (eye(rows) - kalmanGain*model.C)*pvar
  return updatedMean, updatedVar
end

function smooth(kmeans::Array{Float64, 2}, kcovars::Array{Float64, 3}, us::Array{Float64, 2}, model::LLDS{Array{Float64, 2}})
  # Returns the smoothed means and covariances
  # Note, this is only for matrix entries!
  rows, cols = size(kmeans)
  smoothedmeans = zeros(rows, cols)
  smoothedvars = zeros(rows, rows, cols)
  smoothedmeans[:, end] = kmeans[:, end]
  smoothedvars[:, :, end] = kcovars[:, :, end]

  for t=cols-1:-1:2
    Pt = model.A*kcovars[:, :, t]*transpose(model.A) + model.Q
    Jt = kcovars[:, :, t]*transpose(model.A)*inv(Pt)
    smoothedmeans[:, t] = kmeans[:, t] + Jt*(smoothedmeans[:, t+1] - model.A*kmeans[:, t] - model.B*us[:, t-1])
    smoothedvars[:,:, t] = kcovars[:,:, t] + Jt*(smoothedvars[:,:, t+1] - Pt)*transpose(Jt)
  end

  Pt = model.A*kcovars[:, :, 1]*transpose(model.A) + model.Q
  Jt = kcovars[:, :, 1]*transpose(model.A)*inv(Pt)
  smoothedmeans[:, 1] = kmeans[:, 1] + Jt*(smoothedmeans[:, 2] - model.A*kmeans[:, 1]) # no control for the prior
  smoothedvars[:,:, 1] = kcovars[:,:, 1] + Jt*(smoothedvars[:,:, 2] - Pt)*transpose(Jt)


  return smoothedmeans, smoothedvars
end

function predict_visible(kmean::Array{Float64, 1}, kcovar::Array{Float64, 2}, us::Array{Float64, 2}, model::LLDS{Array{Float64, 2}})
  # Predict the visible states n steps into the future given the controller action.
  # Note: us[t] predicts xs[t+1]

  rows, = size(kmean)
  rus, n = size(us)
  predicted_means = zeros(rows, n)
  predicted_covars = zeros(rows, rows, n)

  predicted_means[:, :], predicted_covars[:, :, :] = predict_hidden(kmean, kcovar, us, model)

  rows, cols = size(model.R)
  predicted_vis_means = zeros(rows, n)
  predicted_vis_covars = zeros(rows, cols, n)

  for k=1:n # convert the hidden state to the observed state
    predicted_vis_means[:, k] = model.C*predicted_means[:,k]

    predicted_vis_covars[:, :, k] = model.R + model.C*predicted_covars[:, :, k]*transpose(model.C)
  end

  return predicted_vis_means, predicted_vis_covars
end

function predict_visible(kmean::Array{Float64, 1}, kcovar::Array{Float64, 2}, us::Array{Float64, 1}, model::LLDS{Float64})
  # Predict the visible states n steps into the future given the controller action.
  # Note: us[t] predicts xs[t+1]

  rows, = size(kmean)
  n, = size(us)
  predicted_means = zeros(rows, n)
  predicted_covars = zeros(rows, rows, n)

  predicted_means[:, :], predicted_covars[:, :, :] = predict_hidden(kmean, kcovar, us, model)

  rows, = size(model.R) # actually just the standard deviation
  predicted_vis_means = zeros(rows, n)
  predicted_vis_covars = zeros(rows, n)

  for k=1:n # convert the hidden state to the observed state
    predicted_vis_means[:, k] = model.C*predicted_means[:,k]

    predicted_vis_covars[:, k] = model.R + model.C*predicted_covars[:, :, k]*transpose(model.C)
  end

  return predicted_vis_means, predicted_vis_covars
end

function predict_hidden(kmean::Array{Float64, 1}, kcovar::Array{Float64, 2}, us::Array{Float64, 2}, model::LLDS{Array{Float64, 2}})
  # Predict the hidden states n steps into the future given the controller action.
  # Note: us[t] predicts xs[t+1]

  rows, = size(kmean)
  rus, n = size(us)
  predicted_means = zeros(rows, n)
  predicted_covars = zeros(rows, rows, n)

  predicted_means[:, 1] = model.A*kmean + model.B*us[:,1] + model.b
  predicted_covars[:, :, 1] = model.Q + model.A*kcovar*transpose(model.A)

  for k=2:n #cast the state forward
    predicted_means[:, k], predicted_covars[:, :, k] = step_predict(predicted_means[:,k-1], predicted_covars[:, :, k-1],us[:,k], model)
  end

  return predicted_means, predicted_covars
end

function predict_hidden(kmean::Array{Float64, 1}, kcovar::Array{Float64, 2}, us::Array{Float64,1}, model::LLDS{Float64})
  # Predict the hidden states n steps into the future given the controller action.
  # Note: us[t] predicts xs[t+1]

  rows, = size(kmean)
  n, = size(us)
  predicted_means = zeros(rows, n)
  predicted_covars = zeros(rows, rows, n)

  predicted_means[:, 1] = model.A*kmean + model.B[:, 1]*us[1] + model.b # fix so that Array{Float64, 1}
  predicted_covars[:, :, 1] = model.Q + model.A*kcovar*transpose(model.A)

  for k=2:n #cast the state forward
    predicted_means[:, k], predicted_covars[:, :, k] = step_predict(predicted_means[:,k-1], predicted_covars[:, :, k-1], us[k], model)
  end

  return predicted_means, predicted_covars
end

end #module
