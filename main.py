import numpy as np

G = [[1],[0, 2],[1, 3],[2, 4],[3]] # array at the ith position contains the neighbouring nodes of node i
N = len(G) 
G_params = [0.5]*N # initially unbiased -> Bernoulli parameters
A_V = [0.4 for v in range(N)] # influence of samples -> alpha_v in paper
A_VU = [1/(len(G[v])+1) for v in range(N)] # all neighbours equally influence any node
M = 1 # number of Bernoulli variables
Beta = 0.75  
Lv = 0.167

def generate_samples(T): # for t time steps
    samples = []
    # samples for t=0
    samples.append([])
    for v in range(N):
        sample = []
        for m in range(M):
            sample.append(np.random.binomial(n=1, p=G_params[v]))
        samples[-1].append(sample)        
 
    for t in range(1,T):
        samples.append([])
        # determine the Bernoulli paremeters for t
        for v in range(N):
            influ = 0
            for Nv in G[v]:
                Nv_p = sum(samples[t-1][Nv]) # number of ones
                influ += A_VU[v]*Nv_p/M
            influ += A_VU[v]*samples[t-1][v][0]/M # influenced by self
    
            # updating the Bernoulli parameters
            G_params[v] = (1 - A_V[v])*((1-Beta)*np.random.binomial(n=1,p=0.5) + Beta*Lv) + A_V[v]*influ
            sample = []
            for m in range(M):
                sample.append(np.random.binomial(n=1, p=G_params[v]))
            samples[-1].append(sample) 

    return samples

def generate_samples_coin_toss(d, T, pc): # pc -> coin toss probability
    samples = []
    # samples for t=0
    samples.append([])
    for v in range(N):
        sample = []
        for m in range(M):
            sample.append(np.random.binomial(n=1, p=G_params[v]))
        samples[-1].append(sample)        
 
    for t in range(1,T):
        samples.append([])
        # determine the Bernoulli paremeters for t
        C  = 0
        if t>d: # coin can be tossed 
            C = np.random.binomial(n=1, p=1-pc)    
        for v in range(N):
            influ = 0
            for Nv in G[v]:
                Nv_p = sum(samples[t-1-C*d][Nv]) # number of ones
                influ += A_VU[v]*Nv_p/M
            influ += A_VU[v]*samples[t-1- C*d][v][0]/M # influenced by self
    
            # updating the Bernoulli parameters
            G_params[v] = (1 - A_V[v])*((1-Beta)*np.random.binomial(n=1,p=0.7) + Beta*Lv) + A_V[v]*influ
            sample = []
            for m in range(M):
                sample.append(np.random.binomial(n=1, p=G_params[v]))
            samples[-1].append(sample) 

    return samples
    
def Mu(x):
    return 0.4*x # L = 0.4, Mu_max = 0.4

def generate_samples_random_Mv(T, Mu, M_): # Mu -> L-Lipschitz function, 
    samples = []
    # samples for t=0
    samples.append([])
    for v in range(N):
        sample = []
        M = min(np.random.poisson(Mu(0.5)),M_) + 1
        for m in range(M):
            sample.append(np.random.binomial(n=1, p=G_params[v]))
        samples[-1].append(sample)        
 
    for t in range(1,T):
        samples.append([])
        # determine the Bernoulli paremeters for t
        for v in range(N):
            influ = 0
            for Nv in G[v]:
                Nv_p = sum(samples[t-1][Nv]) # number of ones
                influ += A_VU[v]*Nv_p/len(samples[t-1][Nv])
            influ += A_VU[v]*sum(samples[t-1][v])/len(samples[t-1][v]) # influenced by self
    
            # updating the Bernoulli parameters
            G_params[v] = (1 - A_V[v])*((1-Beta)*np.random.binomial(n=1,p=0.7) + Beta*Lv) + A_V[v]*influ
            sample = []
            M = min(np.random.poisson(Mu(G_params[v])),M_) + 1
            for m in range(M):
                sample.append(np.random.binomial(n=1, p=G_params[v]))
            samples[-1].append(sample) 

    return samples


def generate_samples_r_Mv_r_d(T, Mu, M_, d): # [t-d,t-d+1,....t] -> randomly choose a sample to go forward
    samples = []
    # samples for t=0
    samples.append([])
    for v in range(N):
        sample = []
        M = min(np.random.poisson(Mu(0.5)),M_) + 1
        for m in range(M):
            sample.append(np.random.binomial(n=1, p=G_params[v]))
        samples[-1].append(sample)        
 
    for t in range(1,T):
        samples.append([])
        C = 0
        if t>d:
            C = np.random.randint(0, d)
        # determine the Bernoulli paremeters for t
        for v in range(N):
            influ = 0
            for Nv in G[v]:
                Nv_p = sum(samples[t-1-C][Nv]) # number of ones
                influ += A_VU[v]*Nv_p/len(samples[t-1-C][Nv])
            influ += A_VU[v]*sum(samples[t-1-C][v])/len(samples[t-1-C][v]) # influenced by self
    
            # updating the Bernoulli parameters
            G_params[v] = (1 - A_V[v])*((1-Beta)*np.random.binomial(n=1,p=0.7) + Beta*Lv) + A_V[v]*influ
            sample = []
            M = min(np.random.poisson(Mu(G_params[v])),M_) + 1
            for m in range(M):
                sample.append(np.random.binomial(n=1, p=G_params[v]))
            samples[-1].append(sample)

    return samples      

def generate_samples_r_Mv_coin_toss(T, Mu, M_, d, pc): # [t-d,t-d+1,....t] -> randomly choose a sample to go forward
    samples = []
    # samples for t=0
    samples.append([])
    for v in range(N):
        sample = []
        M = min(np.random.poisson(Mu(0.5)),M_) + 1
        for m in range(M):
            sample.append(np.random.binomial(n=1, p=G_params[v]))
        samples[-1].append(sample)        
 
    for t in range(1,T):
        samples.append([])
        C = 0
        if t>d:
            C = np.random.binomial(n=1, p=1-pc)
        # determine the Bernoulli paremeters for t
        for v in range(N):
            influ = 0
            for Nv in G[v]:
                Nv_p = sum(samples[t-1-C*d][Nv]) # number of ones
                influ += A_VU[v]*Nv_p/len(samples[t-1-C][Nv])
            influ += A_VU[v]*sum(samples[t-1-C][v])/len(samples[t-1-C][v]) # influenced by self
    
            # updating the Bernoulli parameters
            G_params[v] = (1 - A_V[v])*((1-Beta)*np.random.binomial(n=1,p=0.7) + Beta*Lv) + A_V[v]*influ
            sample = []
            M = min(np.random.poisson(Mu(G_params[v])),M_) + 1
            for m in range(M):
                sample.append(np.random.binomial(n=1, p=G_params[v]))
            samples[-1].append(sample)

    return samples  