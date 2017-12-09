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

        alpha[:,0] = self.normalise(self.ep[evidence[0]]*initial)

        for ke in range(1,ne): # loop through each evidence observation
            predictions = numpy.matmul(self.tp, alpha[:,ke-1])
            alpha[:,ke] = self.normalise(self.ep[evidence[ke]]*predictions)

        return alpha

    def backward(self, evidence):
        # Backwards algorithm for hmm.
        ns = len(self.tp) #number of states
        ne = len(evidence) #number of observations
        beta = numpy.zeros([ns, ne]) #matrix of forward probabilities

        #initialise
        beta[:,-1] = 1 # for correct Bayes

        for ke in range(ne-1,0,-1): # iterate backwards over evidence, evidence at t does not matter
            beta[:, ke-1] = self.normalise(numpy.matmul(self.tp, beta[:,ke]*self.ep[evidence[ke]]))
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
