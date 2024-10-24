import networkx as nx
import matplotlib.pyplot as plt
import decision_tree_starter as Trees
import pandas as pd

def recursive_function(n):
    if n <= 0:
        return
    print("hi")
    recursive_function(n - 1)
    return n

titanic_data = pd.read_csv('processed_titanic_training.csv')
titanic_labels = titanic_data['survived']
titanic_ftrs = titanic_data.iloc[:, 2:]
shallow_tree = Trees.DecisionTree(max_depth= 2)
shallow_tree.fit(titanic_ftrs, titanic_labels)

def bfs(q, i):
    #if not queue, return 
    #curr_node = queue.pop
    #mark curr node
    #add all neighbors to queue
    
    if len(q) == 0:
        return 
    tree_node, parent = q.pop(0)
    if tree_node.pred != None: #leaf node
        curr_node = (i, tree_node.pred)
        
    else:
        curr_node = (i, tree_node.split_idx, tree_node.thresh)
    
    G.add_node(curr_node)
    if parent != None:
        G.add_edge(parent, curr_node)
    
    if tree_node.pred == None:    
        q.append((tree_node.left, curr_node))
        # left = bfs(q, parent = )
        q.append((tree_node.right, curr_node))
    bfs(q, i + 1)
    
    
    
    
    return 


def create_call_graph(G, n, parent=None):
    if n <= 0:
        return
    # current_node = f"recursive_function({n})"
    current_node = ((n%3), "hi" )

    G.add_node(current_node)
    if parent:
        G.add_edge(parent, current_node)
    create_call_graph(G, n - 1, current_node)

G = nx.DiGraph()
# create_call_graph(G, 5)
q = [(shallow_tree, None)]
bfs(q, 0)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10)
plt.show()