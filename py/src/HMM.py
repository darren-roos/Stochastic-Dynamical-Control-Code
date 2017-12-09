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


