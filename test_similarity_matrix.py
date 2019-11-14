import numpy as np
from scipy import spatial as spatial

def pair_ind_to_dist_ind(d, i, j):
    index = d*(d-1)/2 - (d-i)*(d-i-1)/2 + j - i - 1
    return index

def dist_ind_to_pair_ind(d, i):
    b = 1 - 2 * d
    x = np.floor((-b - np.sqrt(b**2 - 8*i))/2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    return (x,y)

vecs = np.random.rand(6,4)

similarities = spatial.distance.pdist(vecs)#similar vector

print("vecs = {vec} \n similarities = {sim}\n ".format(vec = vecs,sim = similarities))

similarities_matrix = spatial.distance.squareform(similarities)
print(similarities_matrix )

sim_vecs_sorted = np.argsort(similarities)

print("sorted sim vecs = {0}\n".format(sim_vecs_sorted))

#print(idx,similarities[int(idx)])
#print(np.average(similarities_matrix[0,:]))
#print(np.average(similarities_matrix[:,3]))
#print(np.average(similarities_matrix[3,:]))

k = 0 
pruned_list = []
deleted_conns = []
num_to_be_deleted = 2
out_filters = len(vecs)
for sorted_idx in range(len(sim_vecs_sorted)):
    if k >= num_to_be_deleted: 
        break
                    
    if sim_vecs_sorted[sorted_idx] not in deleted_conns:
        fi,fj = dist_ind_to_pair_ind(out_filters,sim_vecs_sorted[sorted_idx])
        
        print("({0},{1}) is chosed in iter {2}\n".format(fi,fj,sorted_idx))

        fi_degree, fj_degree= np.average(similarities_matrix[fi,:]), np.average(similarities_matrix[:,fj])
        delete_idx = 0
        if fi_degree > fj_degree: #delete the node with the bigger output degree
            pruned_list.append(fi)
            delete_idx = fi
            #refine the weight to the pairs
            #old_weights[fj,...] = old_weights[fj,...]+old_weights[fi,...]
        else:
            pruned_list.append(fj)
            delete_idx = fj
            #old_weights[fi,...] = old_weights[fi,...]+old_weights[fj,...]
        deleted_conns += [int(pair_ind_to_dist_ind(out_filters,delete_idx,d_idx)) for d_idx in range(delete_idx+1,out_filters)]    
        deleted_conns += [int(pair_ind_to_dist_ind(out_filters,d_idx,delete_idx)) for d_idx in range(0,delete_idx)]
        
        k += 1
    print("Deleted filters = {del_conns}".format(del_conns=deleted_conns))

print("pruned list = {0}".format(pruned_list))

import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()

for i in range(len(vecs)):
    for j in range(i+1,len(vecs)):
        G.add_edge(i,j,weight=similarities_matrix[i][j])

position = nx.spring_layout(G)
nx.draw_networkx_nodes(G,position,node_color="r")

#normalize the similarity matrix to (0-1)
similarities_matrix = similarities_matrix/np.max(similarities_matrix)*5
for i in range(len(sim_vecs_sorted)):
    fi,fj = dist_ind_to_pair_ind(out_filters,sim_vecs_sorted[i])
    nx.draw_networkx_edges(G,position,[(fi,fj)],width=int(similarities_matrix[fi][fj]),edge_color='b')
    
#for i in range(len(vecs)):
#    for j in range(i+1,len(vecs)):
#        nx.draw_networkx_edges(G,position,[(i,j)],width=int(similarities_matrix[i][j]),edge_color='b')

for i in range(len(deleted_conns)):
        k,l = dist_ind_to_pair_ind(out_filters,deleted_conns[i])
        nx.draw_networkx_edges(G,position,[(k,l)],width=int(similarities_matrix[k][l]))

#nx.draw_networkx_edges(G,position)
nx.draw_networkx_labels(G,position)
plt.show()
