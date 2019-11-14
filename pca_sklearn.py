import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def mypca(X):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    #assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs[:,0:2])
    return X_pca

#注意如果不用scale将数据归一化到[0-1]的话，sklearn和numpy的结果会不一致
#np.random.seed(0)

X = np.random.rand(4,5)

X = scale(X)

print('original:\n',X)
pca = PCA(n_components=2,svd_solver = 'full',tol=1e-10)

pca.fit(X)


x = pca.transform(X)

print('after PCA:\n',x)

print('singular_values_:\n',pca.singular_values_)
print('components:\n',pca.components_)


h = mypca(X)

print('after def pca :\n',h)

u,s,v = np.linalg.svd(np.transpose(np.asmatrix(X)))
print(u.shape,s.shape,v.shape)
Q = u[:,0:2]
y = np.dot(X,Q)

print('sy:\n',s)
print('vy:\n',Q)
print('after SVD:\n',y)
print(np.argsort(-np.linalg.norm(Q,1,axis=1)))
print(np.linalg.norm(Q,1,axis=1))
#print('ranks of L1-norm',sort)'
'''
u,s,v = np.linalg.svd(X)
idx = np.argsort(-s)
print(u.shape,s.shape,v.shape)
Q = v[:,idx[0:2]]
z = np.dot(X,Q)

print('sz:\n',s)
print('Qz:\n',Q)
print('after SVD2:\n',z)
'''


