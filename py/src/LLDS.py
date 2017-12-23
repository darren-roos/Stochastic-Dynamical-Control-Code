

class llds:
  # Assume zero mean transition and emission defs.
  # The linear latent dynamical system should have the
  # state space form:
  # x(t+1) = A*x(t) + Bu(t) + Q
  # y(t+1) = C*x(t+1) + R (there could be Du(t) term here but we assume the inputs don't affect
  # the measurements directly.)
  # I assume the simplest self I will deal with has matrix A, B and float C therefore
  # the slightly parametric type. Also that specific simple case has only one input.
  # this is to avoid ugly notation later.
    def __init__(A, B, C, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q # Process Noise
        self.R = R # Measurement Noise VARIANCE


    def step(self, xprev, uprev):
        # Controlled, move multivariate self one time step forward.

        xnow = self.A*xprev + self.B*uprev
        ynow = self.C*xnow

        return xnow,  ynow
        
    
    def init_filter(self, initmean, initvar, ynow):
        # Initialise the filter. No prediction step, only a measurement update step.
        updatedMean , updatedVar  = step_update(initmean, initvar, ynow)
        return updatedMean, updatedVar
        

    def step_filter(self, prevmean, prevvar, uprev, ynow):
        # Return the posterior over the current state given the observation and previous
        # filter result.
        pmean , pvar  = step_predict(prevmean, prevvar, uprev)
        updatedMean , updatedVar  = step_update(pmean, pvar, ynow)
        return updatedMean, updatedVar
        
"""
    def step_predict(self, xprev, varprev, uprev):
    # Return the one step ahead predicted mean and covariance.
    pmean = self.A*xprev + self.B*uprev
    pvar =  self.Q + self.A*varprev*transpose(self.A)
    return pmean, pvar
    end

    def step_update(self, pmean, pvar, ymeas):
    # Return the one step ahead measurement updated mean and covar.
    kalmanGain = pvar*transpose(self.C)*inv(self.C*pvar*transpose(self.C) + self.R)
    ypred = self.C*pmean #predicted measurement
    updatedMean = pmean + kalmanGain*(ymeas - ypred)
    rows, cols = len(pvar)
    updatedVar = (eye(rows) - kalmanGain*self.C)*pvar
    return updatedMean, updatedVar
    end

    def smooth(self, kmeans, kcovars, us):
    # Returns the smoothed means and covariances
    # Note, this is only for matrix entries!
    rows, cols = len(kmeans)
    smoothedmeans = numpy.zeros(rows, cols)
    smoothedvars = numpy.zeros(rows, rows, cols)
    smoothedmeans[:, end] = kmeans[:, end]
    smoothedvars[:, :, end] = kcovars[:, :, end]

    for t=cols-1:-1:2
    Pt = self.A*kcovars[:, :, t]*transpose(self.A) + self.Q
    Jt = kcovars[:, :, t]*transpose(self.A)*inv(Pt)
    smoothedmeans[:, t] = kmeans[:, t] + Jt*(smoothedmeans[:, t+1] - self.A*kmeans[:, t] - self.B*us[:, t-1])
    smoothedvars[:,:, t] = kcovars[:,:, t] + Jt*(smoothedvars[:,:, t+1] - Pt)*transpose(Jt)
    end

    Pt = self.A*kcovars[:, :, 1]*transpose(self.A) + self.Q
    Jt = kcovars[:, :, 1]*transpose(self.A)*inv(Pt)
    smoothedmeans[:, 1] = kmeans[:, 1] + Jt*(smoothedmeans[:, 2] - self.A*kmeans[:, 1]) # no control for the prior
    smoothedvars[:,:, 1] = kcovars[:,:, 1] + Jt*(smoothedvars[:,:, 2] - Pt)*transpose(Jt)


    return smoothedmeans, smoothedvars
    end

    def predict_visible(self, kmean, kcovar, us):
    # Predict the visible states n steps into the future given the controller action.
    # Note: us[t] predicts xs[t+1]

    rows, = len(kmean)
    n, = len(us)
    predicted_means = numpy.zeros(rows, n)
    predicted_covars = numpy.zeros(rows, rows, n)

    predicted_means[:, :], predicted_covars[:, :, :] = predict_hidden(kmean, kcovar, us, self)

    rows, = len(self.R) # actually just the standard deviation
    predicted_vis_means = numpy.zeros(rows, n)
    predicted_vis_covars = numpy.zeros(rows, n)

    for k=1:n # convert the hidden state to the observed state
    predicted_vis_means[:, k] = self.C*predicted_means[:,k]

    predicted_vis_covars[:, k] = self.R + self.C*predicted_covars[:, :, k]*transpose(self.C)
    end

    return predicted_vis_means, predicted_vis_covars
    end

    def predict_hidden(self, kmean, kcovar, us):
    # Predict the hidden states n steps into the future given the controller action.
    # Note: us[t] predicts xs[t+1]

    rows, = len(kmean)
    n,  = len(us)
    predicted_means = numpy.zeros(rows, n)
    predicted_covars = numpy.zeros(rows, rows, n)

    predicted_means[:, 1] = self.A*kmean + self.B*us[1]
    predicted_covars[:, :, 1] = self.Q + self.A*kcovar*transpose(self.A)

    for k=2:n #cast the state forward
    predicted_means[:, k], predicted_covars[:, :, k] = step_predict(predicted_means[:,k-1], predicted_covars[:, :, k-1],us[k], self)
    end

    return predicted_means, predicted_covars
    end

    end #module """
