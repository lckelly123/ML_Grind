import torch

x = torch.arange(4.0)
x.requires_grad_(True)

y=2*torch.dot(x,x) #tensor(28., grad_fn=<MulBackward0>)
y.backward() 
#creates a computation graph when going through the operation then the backward commmand 
#traveses that to find derivites at specific points

x.grad  #tensor([ 0.,  4.,  8., 12.])

x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad #tensor([1., 1., 1., 1.])

#non-scalar output:
x.grad.zero_()
y = x*x
y.backward(gradient=torch.ones(len(y)))
x.grad #tensor([0., 2., 4., 6.])


#detaching computations:
x.grad.zero_()
y = x * x
u = y.detach() #this prevents backward from traversing this section of the graph
z = u * x

z.sum().backward()
x.grad == u #true


x.grad.zero_()
y.sum().backward()
x.grad == 2 * x #true