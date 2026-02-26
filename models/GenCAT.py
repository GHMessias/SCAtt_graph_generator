import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import scipy.io

def calc_class_features(S,k,Label):
    pref = np.zeros((len(Label),k))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            pref[nnz[0][i]][Label[nnz[1][i]]] += 1
            pref[nnz[1][i]][Label[nnz[0][i]]] += 1
    for i in range(len(Label)):
        pref[i] /= sum(pref[i])
    pref = np.nan_to_num(pref)

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    # caluculate average and deviation of class preference
    from statistics import mean, median,variance,stdev
    class_pref_mean = np.zeros((k,k))
    class_pref_dev = np.zeros((k,k))
    for i in range(k):
        pref_tmp = []
        for j in partition[i]:
            pref_tmp.append(pref[j])
        pref_tmp = np.array(pref_tmp).transpose()
        for h in range(k):
            class_pref_mean[i,h] = mean(pref_tmp[h])
            class_pref_dev[i,h] = stdev(pref_tmp[h])

    return class_pref_mean, class_pref_dev


def S_class_order(S, n, k, Label):
    import scipy.sparse as sp
    import random
    import copy
    partition = []
    k = max(Label)+1
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    for i in range(k):
        random.shuffle(partition[i])

    community_size = []
    for i in range(len(partition)):
        community_size.append(len(list(partition)[i]))
#     print ("community size : " + str(community_size))
    com_size_dict = {}
    for com_num, size in enumerate(community_size):
        com_size_dict[com_num] = size
    com_size_dict = dict(sorted(com_size_dict.items(), key=lambda x:x[1],  reverse=True))
#     print(com_size_dict)

    communities = copy.deepcopy(partition)
    partition = []
    for com_num in com_size_dict.keys():
        for node in list(communities)[com_num]:
               partition.append(node)
    # print(len(partition))

    import random
    S_class = sp.dok_matrix((n,n))

    part_dic = {}
    for i in range(n):
        part_dic[partition[i]] = i

    nzs = S.nonzero()
    for i in range(len(nzs[0])):
        S_class[part_dic[nzs[0][i]],part_dic[nzs[1][i]]] = 1

    return S_class

def adj_plot(S,Label):
    n=len(Label)
    k=max(Label)+1
    plot_S = S_class_order(S, n, k, Label)
    plt.rcParams["font.size"] = 20
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("node ID", size = 24)
    ax.set_ylabel("node ID", size = 24)
    ax.spy(plot_S, markersize=.2)
    ticks = []
    for _ in range(int(n/1000)+1):
        if len(Label) > 5000 and _ % 2 == 0:
            continue
        ticks.append(_*1000)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.show()
    
def cpm_cpd_plot(S,Label):
    k=max(Label)+1
    import seaborn as sns
    class_pref_mean, class_pref_dev = calc_class_features(S,k,Label)
    plt.rcParams["font.size"] = 13
    plt.title("Class preference mean", fontsize=20)
    hm = sns.heatmap(class_pref_mean,annot=True, cmap='hot_r', fmt="1.2f", cbar=False, square=True)
    plt.xlabel("class",size=20)
    plt.ylabel("class",size=20)
    plt.tight_layout()
    plt.show()
    plt.rcParams["font.size"] = 13
    plt.title("Class preference deviation", fontsize=20)
    hm = sns.heatmap(class_pref_dev,annot=True, cmap='hot_r', fmt="1.2f", cbar=False, square=True)
    plt.xlabel("class",size=20)
    plt.ylabel("class",size=20)
    plt.tight_layout()
    plt.show()
    
def att_plot(X,Label,tag):
    k=max(Label)+1
    plt.rcParams["font.size"] = 21
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1,1,1)
    colors = ['red','blue','green','purple','gold','brown','c','m','k','plum','yellow','pink','maroon','teal','tomato']
    markers = ['.',',','v','^']

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    count = 1
    for i in partition:
        tmp_ver = []
        tmp_hor = []
        for j in i:
            tmp_ver.append(X[j,0])
            tmp_hor.append(X[j,1])
#         ax.scatter(tmp_ver,tmp_hor, c=colors[count],label=count, s=6, marker=markers[count])
        ax.scatter(tmp_ver,tmp_hor, c=colors[count-1],label=count, s=0.05)
        count+=1
        if count == 5: # how many classes do you want to plot?
            break

    plt.xlabel("attribute1", size=32)
    plt.ylabel("attribute2", size=32)
    plt.legend(bbox_to_anchor=(0.45, 1.0), loc='lower center', borderaxespad=1., ncol=4 , markerscale=10., scatterpoints=1, fontsize=18,title='class').get_title().set_fontsize(30)
    plt.tight_layout()
    plt.show()
    
    


def load_data(source):
    """
    Carrega dados de:
    - um caminho (string) para arquivos .npz, .mat, .csv
    - OU um objeto file-like (ex.: io.BytesIO) contendo um .npz.
    """

    # Caso seja um buffer / file-like (BytesIO, file aberto, etc.)
    if hasattr(source, "read"):
        # Vamos assumir que é um .npz em memória
        return load_npz(source)

    # Caso contrário, assume-se que seja uma string com caminho
    if isinstance(source, str):
        ext = source[-3:].lower()
        if ext == "npz":
            return load_npz(source)
        elif ext == "mat":
            return load_mat(source)
        else:
            return load_csv(source)

    raise TypeError(
        f"Tipo de argumento não suportado em load_data: {type(source)}. "
        "Use path (str) ou objeto file-like (BytesIO, arquivo aberto, etc.)."
    )


# def load_npz(file_name):
#     with np.load(file_name, allow_pickle=True) as loader:
#         loader = dict(loader)['arr_0'].item()
#         S = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
#                                               loader['adj_indptr']), shape=loader['adj_shape'])
#         if 'attr_data' in loader:
#             _X_obs = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
#                                                    loader['attr_indptr']), shape=loader['attr_shape'])
#         else:
#             _X_obs = None
#         Label = loader.get('labels')
#     S= S + S.T
#     S[S > 1] = 1
#     lcc = largest_connected_components(S)
#     S = S[lcc,:][:,lcc]
#     Label = Label[lcc]
#     n = S.shape[0]
#     k = len(set(Label))
#     for i in range(n):
#         S[i,i] = 0
#     nonzeros = S.nonzero()
#     m = int(len(nonzeros[0])/2)
#     print ("number of nodes : " + str(n))
#     print ("number of edges : " + str(m))
#     print ("number of classes : " + str(len(set(Label))))
#     return S,Label,n,m,k

def load_npz(source):
    """
    Carrega um .npz em um dos dois formatos:

    1) Formato antigo (usado pelos datasets originais do GenCAT):
       - arquivo contém uma chave 'arr_0' com um dict:
         {
           'adj_data', 'adj_indices', 'adj_indptr', 'adj_shape',
           'attr_data', 'attr_indices', 'attr_indptr', 'attr_shape',
           'labels', ...
         }

    2) Formato novo (usado pelo mimic), mais simples:
       - chaves explícitas:
         - 'edge_index' : array (num_edges, 2)
         - 'x'          : array (n, d)
         - 'y'          : array (n,) ou (n, 1)
    """
    with np.load(source, allow_pickle=True) as loader:

        # -------------------------------
        # CASO 1: formato antigo (arr_0)
        # -------------------------------
        if "arr_0" in loader.files:
            obj = loader["arr_0"].item()
            S = sp.csr_matrix(
                (obj["adj_data"], obj["adj_indices"], obj["adj_indptr"]),
                shape=obj["adj_shape"],
            )
            if "attr_data" in obj:
                _X_obs = sp.csr_matrix(
                    (obj["attr_data"], obj["attr_indices"], obj["attr_indptr"]),
                    shape=obj["attr_shape"],
                )
            else:
                _X_obs = None
            Label = obj.get("labels")

        # -------------------------------
        # CASO 2: formato novo (edge_index, x, y)
        # -------------------------------
        elif {"edge_index", "x", "y"}.issubset(loader.files):
            edge_index = loader["edge_index"]  # esperado shape (num_edges, 2)
            x_arr = loader["x"]
            y_arr = loader["y"]

            # Garantir shapes coerentes
            n = x_arr.shape[0]
            if edge_index.ndim != 2 or edge_index.shape[1] != 2:
                raise ValueError(
                    f"edge_index deve ter shape (num_edges, 2), mas veio {edge_index.shape}"
                )

            rows = edge_index[:, 0].astype(np.int64)
            cols = edge_index[:, 1].astype(np.int64)
            data = np.ones(len(rows), dtype=np.float32)

            # Matriz de adjacência esparsa
            S = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

            # Como no código original: tornamos o grafo não-direcionado,
            # removemos multi-arestas e loops depois
            _X_obs = None
            Label = y_arr

        else:
            raise ValueError(
                "Formato de .npz não reconhecido em load_npz. "
                "Esperado 'arr_0' (formato antigo) ou 'edge_index','x','y' (formato novo)."
            )

    # ---------------------------------
    # Parte comum: pós-processamento
    # (idêntica ao teu load_npz original)
    # ---------------------------------
    S = S + S.T
    S[S > 1] = 1

    lcc = largest_connected_components(S)  # função já usada no teu código
    S = S[lcc, :][:, lcc]
    Label = Label[lcc]

    n = S.shape[0]
    k = len(set(Label))

    # zera diagonal
    for i in range(n):
        S[i, i] = 0

    nonzeros = S.nonzero()
    m = int(len(nonzeros[0]) / 2)

    print("number of nodes : " + str(n))
    print("number of edges : " + str(m))
    print("number of classes : " + str(len(set(Label))))

    return S, Label, n, m, k


def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def load_mat(path): # switch for two form of file
    if "mat" in path:
        print ("mat")
        S,Label,A = for_mat(path)
        nnz = S.nonzero()
        return S,Label,S.shape[0],int(len(nnz[0])/2),len(set(Label))
    
def for_mat(path):
    mat_contents = scipy.io.loadmat(path)
#     print(mat_contents)
    G = mat_contents["S"]
    X = mat_contents["X"]
    Label =np.ndarray.flatten(mat_contents["C"])
    node_size = G.shape[0]
    att_size = X.shape[1]
    S = np.zeros((node_size,node_size))
    if type(X) != np.ndarray:
        A = X.toarray()
    else:
        A = X
    #fill the adjacency matrix and attribute matrix
    nonzeros = G.nonzero()
    print ("no.nodes: " + str(node_size))
    print ("no.attributes: " + str(att_size))
    edgecount=0
    for i in range(len(nonzeros[0])):
        S[nonzeros[0][i],nonzeros[1][i]] = 1
        S[nonzeros[1][i],nonzeros[0][i]] = 1
    # erase diagonal element
    diag = 0
    for i in range(node_size):
#         diag += S[i,i]
        S[i,i] = 0
    return S, Label, A

def load_csv(file_name):
    path = '/Users/seiji/Documents/datasets/factorized-graphs-master/experiments_sigmod20/realData/'
    with open(path+file_name+'-neighbors.csv',mode='r') as f:
        edges = f.read().split('\n')[:-1]
    for i, edge in enumerate(edges):
        edges[i] = edge.split(',')
    with open(path+file_name+'-classes.csv',mode='r') as f:
        classes = f.read().split('\n')[:-1]
    n = len(classes)
    C = np.zeros(n,dtype=int)
    for tmp in classes:
        i,c_i = tmp.split(',')
        C[int(i)] = int(c_i)
    S = sp.lil_matrix((n,n),dtype=int)
    for i,j in edges:
        S[int(i),int(j)] = 1
        S[int(j),int(i)] = 1
    S = S.tocsr()
    return S, C, n, S.sum(), len(set(C))



# GenCAT CORE

import numpy as np
from numpy import linalg as la
from scipy import sparse
from scipy.stats import bernoulli
import random
import copy
import sys
import powerlaw
import warnings
warnings.simplefilter('ignore')

def node_deg(n,m,max_deg):
    p = 3.
    simulated_data = [0]
    while sum(simulated_data)/2 < m:
        theoretical_distribution = powerlaw.Power_Law(xmin = 1., parameters = [p])
        simulated_data=theoretical_distribution.generate_random(n)
        over_list = np.where(simulated_data>max_deg)[0]
        while len(over_list) != 0:
            add_deg = theoretical_distribution.generate_random(len(over_list))
            for i,node_id in enumerate(over_list):
                simulated_data[node_id] = add_deg[i]
            over_list = np.where(simulated_data>max_deg)[0]
        simulated_data = np.round(simulated_data)
        if (m - sum(simulated_data)/2) < m/5:
            p -= 0.01
        else:
            p -= 0.1
        if p<1.01:
            print("break")
            break
    print("expected number of edges : ",sum(simulated_data)/2)
    return sorted(simulated_data,reverse=True)


def count_node_degree(S):
    n = S.shape[0]
    node_degree = np.zeros(n)
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            node_degree[nnz[0][i]] += 1
            node_degree[nnz[1][i]] += 1
    return int(sum(node_degree)/2)

def distribution_generator(flag, para_pow, para_normal, para_zip, t):
    if flag == "power_law":
        dist = 1 - np.random.power(para_pow, t) # R^{k}
    elif flag == "uniform":
        dist = np.random.uniform(0,1,t)
    elif flag == "normal":
        dist = np.random.normal(0.5,para_normal,t)
    elif flag == "zipfian":
        dist = np.random.zipf(para_zip,t)
    return dist

def com_size_gen(k,phi_c):
#     chi = distribution_generator("power_law",phi_c,0,0, k)
    chi = distribution_generator("normal",phi_c,0,0, k)
    return np.array(chi) / sum(chi)
    

def latent_factor_gen(n,k,M,D,com_size):
    density = np.zeros(k)
    for l in range(k):
        density[l] = M[l,l]
    
    # generate U from class preference matrix
    # Freezing function
    def freez_func(q,Th):
        return q**(1/Th) / np.sum(q**(1/Th))
    
    
    U = np.zeros((n,k))
    C=[]
    for i in range(n):
        C_tmp = random.choices(list(range(0,k)),k=1,weights=com_size)[0]
        C.append(C_tmp)
        for h in range(k):
            U[i,h] = np.random.normal(loc=M[C_tmp][h],scale=D[C_tmp][h],size=1)[0]

    # eliminate U<0 and U>1 (keep 0<=U<=1)
    minus_list = np.where(U < 0)
    for i in range(len(minus_list[0])):
        U[minus_list[0][i],minus_list[1][i]] = 0
    one_list = np.where(U > 1)
    for i in range(len(one_list[0])):
        U[one_list[0][i],one_list[1][i]] = 1
    # normalize
    for i in range(n):
        U[i] /= sum(U[i])

    return U,C,density

def adjust(n,k,U,C,M):
    U_prime = copy.deepcopy(U)
    partition = []
    for l in range(k):
        partition.append([])
    for i in range(len(C)):
        partition[C[i]].append(i)
        
    # Freezing function
    def freez_func(q,Th):
        return q**(1/Th) / np.sum(q**(1/Th))
    
    def inverse(U_tmp,l):
        U_ = 1 - U_tmp
        sum_U_ = sum(U_) - U_tmp[l]
        for i in range(k):
            if i != l:
                U_[i] = U_[i] * U_tmp[l] / sum_U_
        return U_
    flag=0
    for l in range(k):
        # Th=1
        loss_min = float('inf')
        if  M[l][l] >= 1/k:
            for Th in np.arange(0.01,1,0.05):
                sum_estimated = np.zeros(k)
                for i in partition[l]:
                    sum_estimated += freez_func(U[i],Th) * freez_func(U[i],Th)
                    # print("freeze_fun", freez_func(U[i],Th) * freez_func(U[i],Th), sum_estimated)
                    if i%150 == 0:
                        break
                # print(M[l],sum_estimated, len(partition[l]))
                loss_tmp = la.norm(M[l]-sum_estimated/len(partition[l]))
                # print("VARIABLE loss_tmp, loss_min", loss_tmp, loss_min)
                if loss_tmp < loss_min:
                    loss_min = loss_tmp
                    Th_min = Th
            for i in partition[l]:
                U[i] = freez_func(U[i],Th_min)
                U_prime[i] = U[i]
        else:
            for Th in np.arange(0.01,1,0.05):
                sum_estimated = np.zeros(k)
                for i in partition[l]:
                    sum_estimated += freez_func(U[i],Th) * inverse(freez_func(U[i],Th),l)
                loss_tmp = la.norm(M[l]-sum_estimated/len(partition[l]))
                if loss_tmp < loss_min:
                    loss_min = loss_tmp
                    Th_min = Th
            for i in partition[l]:
                U[i] = freez_func(U[i],Th_min)
                U_prime[i] = inverse(U[i],l)
#         print(Th_min)
    return U, U_prime
        
def edge_construction(n, U, k, U_prime, step, theta, r):
    U_ = copy.deepcopy(U)
    
    S = sparse.dok_matrix((n,n))
    degree_list = np.zeros(n)
    count_list = []

    print_count = 1
    for i in range(n):
#         if i/n * 10 > print_count:
#             print("finished " +str(print_count)+"0%")
#             print_count += 1
        count = 0
        ng_list = set([i])
        while count < r and degree_list[i] < theta[i]:
            try:
                to_classes = random.choices(list(range(0,k)), k=int(theta[i]-degree_list[i]), weights=U_[i])
            except:
                U_[i] = np.ones_like(U_[i])
                to_classes = random.choices(list(range(0,k)), k=int(theta[i]-degree_list[i]), weights=U_[i])
                print("passed by nan error")
            for to_class in to_classes:
                for loop in range(50):
                    j = U_prime[to_class][int(random.random()/step)]
                    if j not in ng_list:
                        ng_list.add(j)
                        break
                if degree_list[j] < theta[j] and i!=j:
                    S[i,j] = 1;S[j,i] = 1
                    degree_list[i]+=1;degree_list[j]+=1
            count += 1 
        count_list.append(count)
    return S, count_list

def ITS_U_prime(n,k,U_prime,step):
    class_list = []
    UT = U_prime.transpose()
    for i in range(k): # クラスタごとに分布を作成
        UT_tmp = UT[i]/ sum(UT[i])
        for j in range(n-1):
            UT_tmp[j+1] += UT_tmp[j]

        class_tmp = []
        node_counter = 0
        for l in np.arange(0,1,step):
            if node_counter >= n-1:
                class_tmp.append(n-1)
            elif UT_tmp[node_counter] > l:
                class_tmp.append(node_counter)
            else:
                node_counter += 1
                class_tmp.append(node_counter)
        class_list.append(class_tmp)
    return class_list


def adjust_att(n,k,d,U,C,H):
    V = copy.deepcopy(H)
    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(C)):
        partition[C[i]].append(i)
        
    # Freezing function
    def freez_func(q,Th):
        return q**(1/Th) / np.sum(q**(1/Th))
    
    P = np.zeros((k,k))
    for l in range(k):
        for j in partition[l]:
            P[l] += U[j]
        P[l] = P[l]/len(partition[l])
        
    for delta in range(d):
        loss = []
        for Th in np.arange(0.1, 1.1, 0.05):
            loss.append(np.linalg.norm(H[delta] - P @ freez_func(V[delta],Th).T))
        V[delta] = freez_func(V[delta],0.1*(np.argmin(loss)+1))
    return V

def attribute_generation(n,d,k,U,V,C,omega,att_type):
    X = U@V.T

    def variation_attribute(n,d,k,X,C,att_type):
        if att_type == "normal":
            for i in range(d): # each attribute demension
                clus_dev = np.random.uniform(omega,omega,k) # variation for each class
                for p in range(n): # each node
                    X[p,i] += np.random.normal(0.0,clus_dev[C[p]],1)
            # normalization
            for i in range(d):
                X[:,i] -= min(X[:,i])
                X[:,i] /= max(X[:,i])
        else: # Bernoulli distribution
            for i in range(d):
                for p in range(n):
                    X[p,i] = bernoulli.rvs(p=X[p,i], size=1)       
        return X
    return variation_attribute(n,d,k,X,C,att_type)


#####################################################################################
#####################################################################################
########################## modules for ablation study ###############################
#####################################################################################
#####################################################################################

def adjust_woAP(n,k,U,C,density):
    U_prime = copy.deepcopy(U)
    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(C)):
        partition[C[i]].append(i)
        
    def inverse(U_tmp,l,k):
        U_ = 1 - U_tmp
        sum_U_ = sum(U_) - U_tmp[l]
        for i in range(k):
            if i != l:
                U_[i] = U_[i] * U_tmp[l] / sum_U_
        return U_
    for l in range(k):
        if  density[l] < 1/k:
            for j in partition[l]:
                U_prime[i] = inverse(U[j],l,k)
    return U_prime

def edge_construction_wo_ITS(n, U, k, U_primeT, theta, r):
    S = sparse.dok_matrix((n,n))
    degree_list = np.zeros(n)
    count_list = []

    print_count = 1
    reconst = U @ U_primeT
    for i in range(n):
        count = 0
        ng_list = set([i])
       
        while count < r and degree_list[i] < theta[i]:
            to_nodes = random.choices(list(range(0,n)), k=int(theta[i]-degree_list[i]), weights=reconst[i])
            for j in to_nodes:
                if degree_list[j] < theta[j] and i!=j:
                    S[i,j] = 1;S[j,i] = 1
                    degree_list[i]+=1;degree_list[j]+=1
            count += 1 
        count_list.append(count)
    return S, count_list
    

##########################################################################################
##########################################################################################
################################# main function ##########################################
##########################################################################################
##########################################################################################
    
    
def gencat(n,m,k,d,max_deg,M,D,H,phi_c=1,omega=0.2,r=50,step=100,att_type="normal",woAP=False,woITS=False):
    # node degree generation 
    theta = node_deg(n,m,max_deg)
#     line_warn(sum(theta)/2)
    
    # class generation
    com_size = com_size_gen(k,phi_c)
    U,C,density = latent_factor_gen(n,k,M,D,com_size)
    
    # adjusting phase
    if not woAP:
        U,U_prime = adjust(n,k,U,C,M)
    else:
        print("woAP") 
        U_prime = adjust_woAP(n,k,U,C,M)
    
    # Inverse Transform Sampling
    if not woITS:
        step = 1/(n*step)
        U_prime_CDF = ITS_U_prime(n,k,U_prime,step)

        # Edge generation
        S_gen, count_list = edge_construction(n, U, k, U_prime_CDF, step, theta, r)
    else:
        print("woITS")
        S_gen, count_list = edge_construction_wo_ITS(n, U, k, U_prime.T, theta, r)
        
    print("number of generated edges : " + str(count_node_degree(S_gen)))

    V = adjust_att(n,k,d,U,C,H)
    
    # Attribute generation
    X = attribute_generation(n,d,k,U,V,C,omega,att_type)
    
    return S_gen,X,C
    
##########################################################################################
##########################################################################################
############################## for simple input ##########################################
##########################################################################################
##########################################################################################
    
def gencat_simple(n,m,k,d,max_deg,density,H,phi_c=1,omega=0.2,r=50,step=100,att_type="normal"):
    # node degree generation 
    theta = node_deg(n,m,max_deg)
    
    # generate class preference mean from given diagonal elements
    M = np.zeros((k,k))
    for l1 in range(k):
        for l2 in range(k):
            if l1==l2:
                M[l1][l2] = density[l1]
            else:
                M[l1][l2] = (1-density[l1]) / (k-1)
    
    
    # class generation
    U,C = class_generation(n,k,phi_c)
    
    # adjusting phase
    U,U_prime = adjust(n,k,U,C,M)
    
    # Inverse Transform Sampling
    step = 1/(n*step)
    U_prime_CDF = ITS_U_prime(n,k,U_prime,step)

    # Edge generation
    S_gen, count_list = edge_construction(n, U, k, U_prime_CDF, step, theta, r)
    print("number of generated edges : " + str(count_node_degree(S_gen)))
    
    V = adjust_att(n,k,d,U,C,H)
    
    # Attribute generation
    X = attribute_generation(n,d,k,U,V,C,omega,att_type)
    
    return S_gen,X,C

def class_generation(n, k, phi_c):
    com_size = com_size_gen(k,phi_c)
    
    U = np.random.dirichlet(com_size, n)
    C = [] # class assignment list (finally, R^{n})
    for i in range(n):
        C.append(np.argmax(U[i]))

    counter=[];x=[]
    for i in range(k):
        x.append(i)
        counter.append(C.count(i))
    print("class size disribution : ",end="")
    print(counter)
    if 0 in counter:
        print('Error! There is a class which has no member.')
        sys.exit(1)

    return U,C

##########################################################################################
##########################################################################################
############################## for reproduction ##########################################
##########################################################################################
##########################################################################################


def class_reproduction(k,S,Label):
    # extract class preference matrix from given graph
    pref = np.zeros((len(Label),k))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            pref[nnz[0][i]][Label[nnz[1][i]]] += 1
            pref[nnz[1][i]][Label[nnz[0][i]]] += 1
    for i in range(len(Label)):
        pref[i] /= sum(pref[i])

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)
        
    # caluculate average and deviation of class preference
    from statistics import mean, median,variance,stdev
    M = np.zeros((k,k))
    D = np.zeros((k,k))
    for i in range(k):
        pref_tmp = []
        for j in partition[i]:
            pref_tmp.append(pref[j])
        pref_tmp = np.array(pref_tmp).transpose()
        for h in range(k):
            M[i,h] = mean(pref_tmp[h])
            D[i,h] = stdev(pref_tmp[h])
    
    com_size = []
    for i in partition:
        com_size.append(len(i))
    com_size = np.array(com_size) / sum(com_size)
    
    return M,D,com_size
            
def gencat_reproduction(S,Label,H,d,n=0,m=0,max_deg=0,omega=0.2,r=50,step=100,att_type="normal"):

    # node degree generation 
    if n == 0:
        theta = np.zeros(len(Label))
        nnz = S.nonzero()
        for i in range(len(nnz[0])):
            if nnz[0][i] < nnz[1][i]:
                theta[nnz[0][i]] += 1
                theta[nnz[1][i]] += 1
    else:
        theta = node_deg(n,m,max_deg)
    n = len(theta)
    m = count_node_degree(S)
    k = len(set(Label))
    step = 1/(n*step)
    
    # class feature extraction
    M,D,com_size = class_reproduction(k,S,Label)
    
    # latent factor generation
    U,C,density = latent_factor_gen(n,k,M,D,com_size)
    
    # adjusting phase
    U,U_prime = adjust(n,k,U,C,M)
    
    # Inverse Transform Sampling
    U_prime_CDF = ITS_U_prime(n,k,U_prime,step)

    # Edge generation
    S_gen, count_list = edge_construction(n, U, k, U_prime_CDF, step, theta, r)
    print("number of generated edges : " + str(count_node_degree(S_gen)))

    V = adjust_att(n,k,d,U,C,H)
    
    # Attribute generation
    X = attribute_generation(n,d,k,U,V,C,omega,att_type)
    
    return S_gen,X,C


##########################################################################################
##########################################################################################
############################## only attribute ############################################
##########################################################################################
##########################################################################################


def gencat_only_att(n,m,k,d,max_deg,M,D,H,phi_c=1,omega=0.2,r=50,step=100,att_type="normal",woAP=False,woITS=False):
    # node degree generation 
#     theta = node_deg(n,m,max_deg)
#     line_warn(sum(theta)/2)
    
    # class generation
    com_size = com_size_gen(k,phi_c)
    U,C,density = latent_factor_gen(n,k,M,D,com_size)
    
    # adjusting phase
    if not woAP:
        U,U_prime = adjust(n,k,U,C,M)
    else:
        print("woAP") 
        U_prime = adjust_woAP(n,k,U,C,M)
    
    S_gen = []

    V = adjust_att(n,k,d,U,C,H)
    
    # Attribute generation
    X = attribute_generation(n,d,k,U,V,C,omega,att_type)
    
    # not applying user-specified distribution
    X_not = U@V.T
    for i in range(d):
        X_not[:,i] -= min(X_not[:,i])
        X_not[:,i] /= max(X_not[:,i])
    
    return S_gen,X,X_not,C


####################################################
########## BASE GENCAT GENERATOR ###################
####################################################

from core.attributed_graph import AttributedGraph
from core.base_graph_generator import BaseGenerator
import torch
import networkit as nk
import io
import numpy as np
from typing import Optional, Literal


class GenCATGenerator(BaseGenerator):
    def __init__(self):
        super().__init__(name='GenCAT', supports_augment=True, supports_mimic=True)

    def _run_gencat(self, num_nodes,num_edges,num_classes,feature_dim,max_deg,M,D,H,phi_c=1,omega=0.2,r=50,step=100,att_type="normal",woAP=False,woITS=False):
        self.n = num_nodes
        self.m = num_edges
        self.k = num_classes
        self.d = feature_dim
        self.max_deg = max_deg
        self.M = M
        self.D = D
        self.H = H
        self.phi_c = phi_c
        self.omega = omega
        self.r = r
        self.step = step
        self.att_type = att_type
        self.woAP = woAP
        self.woITS = woITS
        S,X,C = gencat(self.n, self.m, self.k, self.d, self.max_deg, 
                       self.M, self.D, self.H, self.phi_c, self.omega, 
                       self.r, self.step, self.att_type, self.woAP, self.woITS)
        
        tmp_graph = nk.Graph(S.shape[0], weighted = False, directed=False)
        _S = S.tocoo()

        for u, v, w in zip(_S.row, _S.col, _S.data):
            if u < v:
                tmp_graph.addEdge(u,v)

        x = x = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(C, dtype=torch.long)

        return AttributedGraph(tmp_graph, x, y)
    

    def generate(self, n, m, k, d, max_deg, M, D, H, phi_c=1, omega=0.2, r=50, step=100, att_type="normal", woAP=False, woITS=False):
        return self._run_gencat(n, m, k, d, max_deg, M, D, H, phi_c, omega, r, step, att_type, woAP, woITS)

    def mimic(self, 
              base_graph: AttributedGraph, 
              num_nodes: Optional[int] = None, 
              num_edges: Optional[int] = None,
              feature_dim: int = 2, 
              H = None, 
              ) -> AttributedGraph:
        """
        Mimicagem dos dados, o objetivo é gerar um novo grafo baseado no grafo presente em `base_graph`
        -----------

        base_graph: AttributedGraph
            Grafo base para a mimicagem. Nele, `base_graph.y` não deve ser None

        n: int
            Quantidade de vértices do grafo clonado. Se `n = None` então clona a quantidade de vértices do grafo original.

        m: int
            Quantidade de vértices do grafo  clonado. Se `m = None` mantém a mesma densidade do grafo original para o valor `n` de vértices.

        d: int
            Quantidade de features de cada vértice

        H: np.array
            Relação entre features e as classes (Detalhes disponíveis no paper GenCAT).
        """
        
        buffer = io.BytesIO()

        edges = []
        for (u, v) in base_graph.graph.iterEdges():
            if u < v:
                edges.append([u, v])

        np.savez(
                buffer,
                edge_index=np.array(edges),
                x=base_graph.x.detach().numpy(),
                y=base_graph.y.detach().numpy()
                )
        
        # Voltar para o início do arquivo
        buffer.seek(0)

        # Ler com o GenCAT, se load_data aceitar file-like object
        S_ori, C, _, _, k = load_data(buffer)

        if num_nodes is None:
            num_nodes = base_graph.num_nodes()

        if num_edges is None:
            density = nk.graphtools.density(base_graph.graph)
            num_edges = int(density * (num_nodes * (num_nodes-1))) / 2

        theta = np.zeros(len(C))
        nnz = S_ori.nonzero()
        for i in range(len(nnz[0])):
            if nnz[0][i] < nnz[1][i]:
                theta[nnz[0][i]] += 1
                theta[nnz[1][i]] += 1
        maxdeg = max(theta)
        
        if H is None:
            H = np.random.rand(feature_dim, base_graph.num_classes())
        S,X,Label = gencat_reproduction(S = S_ori, Label = C, H = H, d = feature_dim, n=num_nodes,m=num_edges,max_deg=int(maxdeg*2))
        tmp_graph = nk.Graph(num_nodes, weighted = False, directed = False)
        _S = S.tocoo()

        for u, v, w in zip(_S.row, _S.col, _S.data):
            if u < v:
                tmp_graph.addEdge(u,v)

        x = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(Label, dtype=torch.long)

        return AttributedGraph(tmp_graph, x, y)