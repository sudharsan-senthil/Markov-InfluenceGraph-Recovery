import copy
from estimators import H_v_plus_C_Q, H_v_plus_C_Q_past

def RecGreedy_0(E, S): # E-> epsilon
    N = len(S[0])
    N_V = []
    for v in range(N):
        N_v ,T = [v], []   # T -> is a temporary neighbour set
        iterate_1 = True
        while iterate_1:
            # print("iteration 1")
            T = copy.deepcopy(N_v) # temp neighbour nodes list
            iterate_2 = True
            T_C = [x for x in range(N) if x not in T] # complement of T
            last = -1
            while iterate_2: # to add the last node appended to T as a legit neighbour node 
                # print("iteration 2")
                if not len(T_C):
                    iterate_1 = False
                    break

                # find out the node, that reduces the conditional entropy the most
                min_H_v_Q, min_T_C_v = 10**5, -1
                for T_C_v in T_C:
                    h = H_v_plus_C_Q(v, T+ [T_C_v], S)                    
                    if h < min_H_v_Q:
                        min_H_v_Q = h
                        min_T_C_v = T_C_v

                if H_v_plus_C_Q(v, T, S) - min_H_v_Q > E/2:
                    T.append(min_T_C_v)
                    T_C.remove(min_T_C_v)
                    last = min_T_C_v
                else:
                    iterate_2 = False
                    if last != -1:
                        N_v.append(last)
                    else:
                        iterate_1 = False # no new node is appended to T, hence all legit neighbour nodes are added to N_v
                        N_V.append(N_v)

        print("neighbours of node v", N_v)

    return N_V

def RecGreedy_1(E,S,D): # for the coin toss, Y(t) = {y(t), y(t-d)} --> Markov chain
    N = len(S[0])
    N_V = []
    for v in range(N):
        N_v ,T = [v], []   # T -> is a temporary neighbour set
        iterate_1 = True
        while iterate_1:
            # print("iteration 1")
            T = copy.deepcopy(N_v) # temp neighbour nodes list
            iterate_2 = True
            T_C = [x for x in range(N) if x not in T] # complement of T
            last = -1
            while iterate_2: # to add the last node appended to T as a legit neighbour node 
                # print("iteration 2")
                if not len(T_C):
                    iterate_1 = False
                    break

                # find out the node, that reduces the conditional entropy the most
                min_H_v_Q, min_T_C_v = 10**5, -1
                for T_C_v in T_C:
                    h = H_v_plus_C_Q_past(v, T+ [T_C_v], D, S)                    
                    if h < min_H_v_Q:
                        min_H_v_Q = h
                        min_T_C_v = T_C_v

                if H_v_plus_C_Q_past(v, T, D, S) - min_H_v_Q > E/2:
                    T.append(min_T_C_v)
                    T_C.remove(min_T_C_v)
                    last = min_T_C_v
                else:
                    iterate_2 = False
                    if last != -1:
                        N_v.append(last)
                    else:
                        iterate_1 = False # no new node is appended to T, hence all legit neighbour nodes are added to N_v
                        N_V.append(N_v)

        print("neighbours of node v", N_v)

    return N_V