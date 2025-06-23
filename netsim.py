import numpy as np

M = 1 # number of Bernoulli variables
Beta = 0.75  
Lv = 0.167

def process_graph(g):
    N = len(g) 
    G_params = [0.5]*N # initially unbiased -> Bernoulli parameters
    A_V = [0.8 for v in range(N)] # influence of samples -> alpha_v in paper
    A_VU = [1/(len(g[v])+1) for v in range(N)] # all neighbours equally influence any node
    return N, G_params, A_V, A_VU

def generate_samples_coin_toss(G, T, Mu, M_, d, pc): # paper's , coin tosses throughout the process
    N, G_params, A_V, A_VU = process_graph(G)
    samples = []
    tail_counts = 0
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
            C = np.random.binomial(n=1, p=(1-pc))
            tail_counts += C
            
        # determine the Bernoulli paremeters for t
        for v in range(N):
            influ = 0
            for Nv in G[v]:
                Nv_p = sum(samples[t-1-C*d][Nv]) # number of ones
                influ += A_VU[v]*Nv_p/len(samples[t-1-C*d][Nv])
            influ += A_VU[v]*sum(samples[t-1-C*d][v])/len(samples[t-1-C*d][v]) # influenced by self
    
            # updating the Bernoulli parameters
            G_params[v] = (1 - A_V[v])*((1-Beta)*np.random.binomial(n=1,p=0.7) + Beta*Lv) + A_V[v]*influ
            sample = []
            M = min(np.random.poisson(Mu(G_params[v])),M_) + 1
            for m in range(M):
                sample.append(np.random.binomial(n=1, p=G_params[v]))
            samples[-1].append(sample)
    # print("# tails = ", tail_counts)
    return samples

def generate_samples_r_d(G, T, Mu, M_, d): # [t-d,t-d+1,....t] -> randomly choose a sample to go forward
    N, G_params, A_V, A_VU = process_graph(G)

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
            C = np.random.randint(0, d+1)
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

def generate_samples_coin_toss_subset(G, T, Mu, M_, d, pc, N1):  # N1 is a list of nodes that toss coin
    N, G_params, A_V, A_VU = process_graph(G)
    
    N0 = [x for x in list(range(N)) if x not in N1]
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
            C = np.random.binomial(n=1, p=(1-pc))
        # determine the Bernoulli paremeters for t
        for v in N1:
            influ = 0
            for Nv in G[v]:
                Nv_p = sum(samples[t-1-C*d][Nv]) # number of ones
                influ += A_VU[v]*Nv_p/len(samples[t-1-C*d][Nv])
            influ += A_VU[v]*sum(samples[t-1-C*d][v])/len(samples[t-1-C*d][v]) # influenced by self

            # updating the Bernoulli parameters
            G_params[v] = (1 - A_V[v])*((1-Beta)*np.random.binomial(n=1,p=0.7) + Beta*Lv) + A_V[v]*influ
            sample = []
            M = min(np.random.poisson(Mu(G_params[v])),M_) + 1
            for m in range(M):
                sample.append(np.random.binomial(n=1, p=G_params[v]))
            samples[-1].append(sample)


        for v in N0:
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

def generate_samples_coin_toss_1(G, T, Mu, M_, d, pc):  # no coin tosses for the next d turns after a tail
    N, G_params, A_V, A_VU = process_graph(G)
    
    samples = []
    tail_counts = 0
    trck = 0 # to keep track of the no toss counts
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
        if t>d and trck==0:
            C = np.random.binomial(n=1, p=(1-pc))
            tail_counts += C  # to make sure no coin tosses for next d turns after a tail
            if C:
                trck = d
        elif trck:
            trck -= 1
            
        # determine the Bernoulli paremeters for t
        for v in range(N):
            influ = 0
            for Nv in G[v]:
                Nv_p = sum(samples[t-1-C*d][Nv]) # number of ones
                influ += A_VU[v]*Nv_p/len(samples[t-1-C*d][Nv])
            influ += A_VU[v]*sum(samples[t-1-C*d][v])/len(samples[t-1-C*d][v]) # influenced by self
    
            # updating the Bernoulli parameters
            G_params[v] = (1 - A_V[v])*((1-Beta)*np.random.binomial(n=1,p=0.7) + Beta*Lv) + A_V[v]*influ
            sample = []
            M = min(np.random.poisson(Mu(G_params[v])),M_) + 1
            for m in range(M):
                sample.append(np.random.binomial(n=1, p=G_params[v]))
            samples[-1].append(sample)
    # print("# tails = ", tail_counts)
    return samples