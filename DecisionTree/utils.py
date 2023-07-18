import pandas,numpy
from treelib import Tree
def split_var(
            var,
            target,
            task,
            cost
            ):
    n,      = var.shape
    #sorting
    var    = var.sort_values()
    invidx = (idx:=var.index).tolist()[::-1]
    if task=='classification':
        target = pandas.get_dummies(target.loc[idx]).cumsum()

        #dropping duplicates
        target = target.assign(var=var).groupby('var').transform('max')[target.columns].drop_duplicates()

        #sides
        left   = target.astype('float')
        right  = left.max() - left
        
        #samples sizes
        n_left   = left.sum(axis=1)
        n_right  = right.sum(axis=1)
    elif task=='regression':
        yhat       = target.loc[idx].cumsum()/(ns:=list(range(1,n+1)))
        left       = pandas.DataFrame([{'ys':target.loc[idx].iloc[:ix],'yhat':y} for (ix,y) in zip(ns,yhat)],index=idx)        
        right      = left.loc[invidx]
        right.index=idx
        n_right    = n-(n_left:=pandas.Series(ns,index=idx))
        
    #cost
    
    cost_left  = cost(n_left,left)
    cost_right = cost(n_right,right)
    J          = (n_left/n)*cost_left+(n_right/n)*cost_right
    
    #optimum
    optix         = J.sort_values().head(1).index.values[0]
    
    
    return (
                (var.name,var.loc[optix],J.loc[optix]) ,(
                    
                        {'sample': n_left.loc[optix] ,'class' : left.loc[optix].to_dict(), 'cost':cost_left.loc[optix]},
                        {'sample': n_right.loc[optix],'class' : right.loc[optix].to_dict(),'cost':cost_right.loc[optix]},

    
            )
    )

def split_node(X,y,task,cost):
    return sorted(
                    [

                    split_var(X[var],y,task,cost)
                    
                    for var in X

                ],key=lambda tr : tr[0][-1] )[0]

def get_node(ix,ix0,X,y,split,task):
    
    ((var,tr,g),(linfo,rinfo)) = split
    
    lmask       = lambda d : d[var]<=tr
    rmask       = lambda d : ~lmask(d)
    
    idx        = lambda left,right : [((ix,0),left),((ix,1),right)] 
    output     = lambda side : side['class']['yhat'] if task=='regression' else max(side['class'].items(),key=lambda x : x[1])[0]
    
    return idx(ix0,ix0),\
           idx(output(linfo),output(rinfo)),\
           idx(linfo,rinfo),\
           idx(lmask,rmask),\
           idx(((X[(lm:=lmask(X))],y[lm]),linfo['sample']),((X[(rm:=rmask(X))],y[rm]),rinfo['sample']))
                                                              
### Cost functions
gini = lambda n,freq : 1 - (freq.div(n,axis=0)**2).sum(axis=1)     
mse  = lambda n,side : pandas.Series([ ((y-yhat)**2).mean() for y,yhat  in side.values],index=side.index)/n    




def grow_tree(X,y,task='classification',cost=gini,min_sample_leaf=1,max_depth=2):
    masks = []
    #depth 0
    ix,ix0 = (0,0),((-1,-1),-1)
    links,classes,infos,masksd,Xy0s        = get_node(ix,ix0,X,y,split_node(X,y,task,cost),task)
    
    masks.append(masksd)
    
    for d in range(1,max_depth):
        Xys1     = []
        masksd   = []
        for i,(ix0,((X0,y0),n_samples)) in enumerate(Xy0s):
            
            if n_samples > min_sample_leaf:
                
                
                link,classe,info,mask,Xys  = get_node((d,i),ix0,X0,y0,split_node(X0,y0,task,cost),task)
                
                links.extend(link)
                classes.extend(classe)
                infos.extend(info)
                masksd.extend(mask)
                Xys1.extend(Xys)
                
        masks.append(masksd)        
        Xy0s = Xys1
                       
    
    return links,classes,masks,infos

def predict(X,links,classes,masks):
    
    links,classes = dict(links),dict(classes)
    
    tests         = [ pandas.DataFrame({node:mask(X) for (node,mask) in depth_masks}) for depth_masks in masks]
    
    instances = []

    depth_instances = {}
    depth = tests[0]
    for node in depth:
            depth_instances[node]=depth[depth[node]].index 
    instances.append(depth_instances)

    for d,depth in enumerate(tests[1:],1) :
        depth_instances = {}
        for node in depth:
            previous = links[node]
            previous_node_instances = instances[d-1][previous]
            node_instances          = depth.loc[previous_node_instances][
                                      depth.loc[previous_node_instances][node]
                                     ].index
            depth_instances[node] = node_instances
            instances[d-1][previous] = instances[d-1][previous].drop(node_instances)
        instances.append(depth_instances)
    index_classes = pandas.DataFrame([ (index,node)   for depth in instances for node in depth for index in depth[node]],
                              columns=['index','node']
                              ).assign(
                        classes = lambda Df : Df.node.map(classes)
                    ).set_index('index').classes.loc[X.index]
    return index_classes
def daw_tree(links):
    

    tree = Tree()
    tree.create_node(str(links[0][1]),str(links[0][1]))
    for (target,source) in links :
        tree.create_node(str(target),  str(target)   , parent=str(source))
    tree.show()