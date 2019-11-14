# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import networkx as nx
import numpy as np
import itertools

import matplotlib.pyplot as plt

# +
G = nx.karate_club_graph()

print('#nodes:', len(G.nodes()), 'and', '#edges:', len(G.edges()))

# +
node_to_degrees = sorted(dict(G.degree()).items(), key = lambda x: x[1], reverse = True)

## view the top 5,
node_to_degrees[:5]

# +
## calculate degree centrality,
closeness_centrality = nx.closeness_centrality(G)
print(closeness_centrality)
## set degree centrality metrics on each node,
nx.set_node_attributes(G, closeness_centrality, 'cc')

sorted(G.nodes(data=True), key = lambda x: x[1]['cc'], reverse=True)

# +
nodes = G.nodes(data = True)

clubs = np.unique([ data['club'] for n, data in nodes ])


plt.figure(figsize=(8, 8))

layout = nx.spring_layout(G)

colors = [ 'r', 'c' ]
for i, club in enumerate(clubs):
    color = colors[i]
    
    nodes_for_club = [ (n, data) for n, data in nodes if data['club'] == club ]
    print('nodes_for_club:',nodes_for_club)
    print('sorted:',itertools.groupby(sorted(nodes_for_club, key=lambda x: x[1]['cc']), key=lambda x: x[1]['cc']))
    for key, nodes_in_club in itertools.groupby(sorted(nodes_for_club, key=lambda x: x[1]['cc']), key=lambda x: x[1]['cc']):
        nodelist = [ n[0] for n in nodes_in_club ]
        nx.draw_networkx_nodes(
            G,
            layout,
            nodelist=nodelist,
            node_color=color,
            node_size=key*500)

nx.draw_networkx_labels(G, layout, dict([ (n[0], n[0]) for n in nodes ]), font_size=5)
nx.draw_networkx_edges(G, layout, width=1.0, alpha=0.5)

plt.axis('off')
# -


