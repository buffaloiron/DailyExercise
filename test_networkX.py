import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

sim_matrix = np.random.rand(4,4)

G=nx.Graph()
for i in range(0,4):
    for j in range(i+1,4):
        G.add_edge(i,j,weight=sim_matrix[i][j])
'''
#加入带权边
G.add_edge(1,2,weight=0.6)
G.add_edge(1,3,weight=0.2)
G.add_edge(3,4,weight=0.1)
G.add_edge(3,5,weight=0.7)
G.add_edge(3,6,weight=0.9)
G.add_edge(1,7,weight=0.3)
'''

#按权重划分为重权值得边和轻权值的边
#elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
#esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.5]
#节点位置
pos=nx.spring_layout(G) # positions for all nodes
#首先画出节点位置
# nodes
nx.draw_networkx_nodes(G,pos,node_size=700)
#依据权重，实线为权值大的边。虚线为权值小的边
# edges
nx.draw_networkx_edges(G,pos,width=6)
#nx.draw_networkx_edges(G,pos,edgelist=esmall,
#                            width=6,alpha=0.5,edge_color='b',style='dashed')

# labels标签定义
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display
