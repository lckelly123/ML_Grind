import torch


#makes a vector of evenly spaced values
x=torch.arange(12, dtype=torch.float32) 

x.numel() #=12

x.shape #torch.Size([12])

X = x.reshape(3,4) #change shape of tensory ans: tensor([[ 0.,  1.,  2.,  3.],[ 4.,  5.,  6.,  7.], [ 8.,  9., 10., 11.]])

torch.zeros((2,3,4)) #creates a 3d tensor filled w zeros.

#INDEXING: X[outermost dim, 2ndmost, ... row, col]
#SPLICING: X[start : stop : step]

X[1, 2] = 17 #indexs and writes

X[:2, :] = 12 #sets top 2 rows to 12


x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

X == Y #tensor([[False,  True, False,  True],[False, False, False, False],[False, False, False, False]])

#BROADCASTING: perform elementwise operations of tensors w differnt shapes

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

a + b #tensor([[0, 1],[1, 2],[2, 3]])

before = id(Y)
Y = Y + X
id(Y) == before #false, python first evals Y+X, then points Y to new mem

#this is an issue as in ML often need to update many params multiple times a second,
#also the same params are often refered by many spots, so could cause stale param issue

#[:] gets around this by overwriting in place

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z)) #id(Z): 140381179266448, id(Z): 140381179266448. IN PLACE!

#converting between python objects:
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B) #(numpy.ndarray, torch.Tensor)


