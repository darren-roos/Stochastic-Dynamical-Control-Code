import numpy

class HMM:
    def __init__(self, tp, ep):
        self.tp = tp
        self.ep = ep

    def normalise(self, vec):
      # Normalise mat such that sum(vec) = 1
        return vec/sum(vec)

    def forward(self, initial, evidence):
        # Forwards algorithm for hmm
        ns = len(self.tp) #number of states
        ne = len(evidence) #number of observations
        alpha = numpy.zeros([ns, ne]) #matrix of forward probabilities

        for ks in range(ns): # first observation
            alpha[ks][0] = self.ep[evidence[0]][ks]*initial[ks]
        
        alpha[:,0] = self.normalise(alpha[:,0]) # normalise probabilities

        for ke in range(1,ne): # loop through each evidence observation

            for ks in range(ns): # loop through each state to build up the joint given observations

                prediction = 0 # the prediction part given the old joint
                for ks_pred in range(ns):
                    prediction += self.tp[ks][ks_pred]*alpha[ks_pred][ke-1] # can be made faster by looping column major!

                alpha[ks][ke] = self.ep[evidence[ke]][ks]*prediction

            alpha[:, ke] = self.normalise(alpha[:,ke]) # normalise probabilities

        return alpha

    def backward(self, evidence):
        # Backwards algorithm for hmm.
        ns = len(self.tp) #number of states
        ne = len(evidence) #number of observations
        beta = numpy.zeros([ns, ne]) #matrix of forward probabilities

        #initialise
        beta[:,-1] = [1.0]*ns # for correct Bayes

        for ke in range(ne-1,0,-1): # iterate backwards over evidence, evidence at t does not matter

            for ks in range(ns): # iterate over states
                recur = 0
                for ks_next in range(ns): #sum over next state
                    recur += self.ep[evidence[ke]][ks_next]*self.tp[ks_next][ks]*beta[ks_next][ke]

                beta[ks][ke-1] = recur

            beta[:, ke-1] = self.normalise(beta[:, ke-1])
        return beta


    def smooth(self, initial, evidence, timeLocation):
        # Forwards-Backwards algorithm. Note that it is required to split the evidence
        # accordingly.
        forwardEvidence = evidence[0:timeLocation]
        backwardEvidence = evidence[timeLocation:]

        alpha = self.forward(initial, forwardEvidence)[:, -1]
        beta = self.backward(backwardEvidence)[:,0]

        smoothed = self.normalise(alpha*beta)

        return smoothed
