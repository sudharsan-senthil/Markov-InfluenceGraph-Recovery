from collections import Counter
import math

def P_estimator(S): # where S is a 3d array, containing samples from T time instances, and every instance contains N*M binary samples
    T, N = len(S), len(S[0])
    S1 = [] # contains elements from the samplespace
    for t in range(T):
        S1.append([])
        for v in range(N):
            S1[-1].append(sum(S[t][v])/len(S[t][v]))
    
    S2 = [[x[v] for x in S1] for v in range(N)] # each array contains samples of one particular node
    P_V = []
    for v in range(N):
        counts = Counter(S2[v])
        P_V.append({key: value/T for key, value in counts.items()})

    return P_V

def P_v_plus_Q(v_plus, Q, S): # v_plus-> node-v value @ (t+1), Q is a subset of nodes [include node-v here if required]
    T, N = len(S), len(S[0])
    S1 = []
    for t in range(T):
        S1.append([])
        for v in range(N):
            S1[-1].append(sum(S[t][v])/len(S[t][v]))
    S2 = []
    for t in range(T-1):
        sample = tuple([S1[t+1][v_plus]] + [S1[t][v] for v in Q]) # tuples is required for hashing
        S2.append(sample)

    counts = Counter(S2)
    P = {key: value/T for key, value in counts.items()} # P([V(t+1),Q(t)])

    return P      

def P_Q(Q, S): # Q includes v
    T, N = len(S), len(S[0])
    S1 = []
    for t in range(T):
        S1.append([])
        for v in range(N):
            S1[-1].append(sum(S[t][v])/len(S[t][v]))
    S2 = []
    for t in range(T-1):
        sample = tuple([S1[t][v] for v in Q]) # tuples is required for hashing
        S2.append(sample)

    counts = Counter(S2)
    P = {key: value/T for key, value in counts.items()} # P(Q(t))

    return P

def H(prob_dist):
    return -sum(p * math.log2(p) for p in prob_dist if p > 0)

def H_v_plus_C_Q(v_plus, Q, S): # Q includes v
    Pr_v_plus_Q = P_v_plus_Q(v_plus, Q, S).values()
    Pr_Q = P_Q(Q, S).values()
    
    return H(Pr_v_plus_Q) - H(Pr_Q) 


def P_Q_past(Q, D, S): # D is a list of time stamps. for coin-toss model D = [0,d]. P({Q(t),Q(t-d)})
    T, N = len(S), len(S[0])
    S1 = []
    for t in range(T):
        S1.append([])
        for v in range(N):
            S1[-1].append(sum(S[t][v])/len(S[t][v]))
    S2 = []
    for t in range(D[-1],T-1):
        sample = []
        for d in D:
            sample += [S1[t-d][v] for v in Q]
        sample = tuple(sample) # tuples is required for hashing
        S2.append(sample)

    counts = Counter(S2)
    P = {key: value/T for key, value in counts.items()} # P(Q(t))

    return P

def P_v_plus_Q_past(v_plus, Q, D, S):
    T, N = len(S), len(S[0])
    S1 = []
    for t in range(T):
        S1.append([])
        for v in range(N):
            S1[-1].append(sum(S[t][v])/len(S[t][v]))
    S2 = []
    for t in range(D[-1],T-1):
        sample = []
        for d in D:
            sample += [S1[t-d][v] for v in Q]
        sample = tuple([S1[t+1][v_plus]] + sample) # tuples is required for hashing
        S2.append(sample)

    counts = Counter(S2)
    P = {key: value/T for key, value in counts.items()} # P(Q(t))

    return P



def H_v_plus_C_Q_past(v_plus, Q, d, S): # H(v+ |{Q(t),Q(t-d)})
    Pr_v_plus_Q_past = P_v_plus_Q_past(v_plus, Q, S).values()
    Pr_Q_past = P_Q_past(Q, S).values()
    
    return H(Pr_v_plus_Q_past) - H(Pr_Q_past)