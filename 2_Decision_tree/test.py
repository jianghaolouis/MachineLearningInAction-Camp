import decision_tree
import numpy as np
tree1 = decision_tree.dt()
tree1.get_dataset([[1,1,'1'],[1,1,'1'],[1,0,'0'],[0,1,'0'],[0,0,'0']])
labels = ['no surfacing','flippers']
print('the entropy of the above dataset is %f' % tree1.ent(tree1.dataset))    #%d %f %s
dt=tree1.split(tree1.dataset,0,1)
print(dt)