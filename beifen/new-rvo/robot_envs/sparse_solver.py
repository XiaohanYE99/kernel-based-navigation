import torch
print('PyTorch version:',torch.__version__)
import time
torchdevice = torch.device('cpu')
if torch.cuda.is_available():
  torchdevice = torch.device('cuda')
  print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
print('Running on ' + str(torchdevice))

# PyTorch also doesn't support any sparse solver
# -> workaround by using cupy with a custom backward pass
# https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html#solving-linear-problems
# https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
# Note that the default cupy version clashes with the default pytorch version
# in colab -> use cupy 10

import cupy as cp
import cupyx.scipy.sparse.linalg
print('CuPy version:',cp.__version__)
print('Running on ',cp.array([1]).data.device)

# Convenience function to map a torch COO tensor in a cupy one
def coo_torch2cupy(A):
  A = A.data.coalesce()
  Avals_cp = cp.asarray(A.values())
  Aidx_cp = cp.asarray(A.indices())
  return cp.sparse.coo_matrix((Avals_cp, Aidx_cp))

# Custom PyTorch sparse solver exploiting a CuPy backend
# See https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
class SparseSolve(torch.autograd.Function):
  @staticmethod
  def forward(ctx, A, b):
    # Sanity check
    if A.ndim != 2 or (A.shape[0] != A.shape[1]):
      raise ValueError("A should be a square 2D matrix.")
    # Transfer data to CuPy
    
    A_cp = coo_torch2cupy(A)
    
    b_cp = cp.asarray(b.data)
    t0=time.time()
    # Solver the sparse system
    if (b.ndim == 1) or (A.shape[1] == 1):
      # cp.sparse.linalg.spsolve only works if b is a vector but is fully on GPU
      x_cp = cp.sparse.linalg.spsolve(A_cp, b_cp)
    else:
      # Make use of a factorisation (only the solver is then on the GPU)
      factorisedsolver = cp.sparse.linalg.factorized(A_cp)
      x_cp = factorisedsolver(b_cp)
    # Transfer (dense) result back to PyTorch
    #print(time.time()-t0)
    x = torch.as_tensor(x_cp, device=torchdevice)
    # Not sure if the following is needed / helpful
    #if A.requires_grad or b.requires_grad:
    x.requires_grad = True
    # Save context for backward pass
    ctx.save_for_backward(A, b, x)
    return x

  @staticmethod
  def backward(ctx, grad):
    # Recover context
    A, b, x = ctx.saved_tensors
    # Compute gradient with respect to b
    gradb = SparseSolve.apply(A.t(), grad)
    # The gradient with respect to the (dense) matrix A would be something like
    # -gradb @ x.T but we are only interested in the gradient with respect to
    # the (non-zero) values of A
    gradAidx = A.indices()
    mgradbselect = -gradb.index_select(0,gradAidx[0,:])
    xselect = x.index_select(0,gradAidx[1,:])
    mgbx = mgradbselect * xselect
    if x.dim() == 1:
      gradAvals = mgbx
    else:
      gradAvals = torch.sum( mgbx, dim=1 )
    gradAs = torch.sparse_coo_tensor(gradAidx, gradAvals, A.shape)
    return gradAs, gradb
'''
sparsesolve = SparseSolve.apply

# Test matrix-vector solver
Aref = torch.eye(10000,10000, dtype=torch.float64, requires_grad=False, device=torchdevice).to_sparse()
Aref.requires_grad=False
bref = torch.eye(10000,1, dtype=torch.float64, requires_grad=False, device=torchdevice)

A = Aref.detach().clone().requires_grad_(True)
b = bref.detach().clone().requires_grad_(True)
t0=time.time()
# Solve
x = sparsesolve(A,b)
#print(time.time()-t0)
# random scalar function to mimick a loss
#loss = x.sum()
#loss.backward()

print('x',x)
with torch.no_grad(): print('allclose:',torch.allclose(A @ x, b))
print('A.grad',A.grad)
print('b.grad',b.grad)

# Compare with dense op
A = Aref.detach().clone().to_dense().requires_grad_(True)
b = bref.detach().clone().requires_grad_(True)
t0=time.time()
x = torch.linalg.solve(A,b)
#print(time.time()-t0)
loss = x.sum()
loss.backward()
print('x',x)
with torch.no_grad(): print('allclose:',torch.allclose(A @ x, b))
print('A.grad',A.grad)
print('b.grad',b.grad)

# Test matrix-matrix solver
Aref = torch.randn(3,3, dtype=torch.float64, requires_grad=False, device=torchdevice).to_sparse()
Aref.requires_grad=False
bref = torch.randn(3,2, dtype=torch.float64, requires_grad=False, device=torchdevice)

A = Aref.detach().clone().requires_grad_(True)
b = bref.detach().clone().requires_grad_(True)

# Solve
x = sparsesolve(A,b)

# random scalar function to mimick a loss
loss = x.sum()
loss.backward()

print('x',x)
with torch.no_grad(): print('allclose:',torch.allclose(A @ x, b))
print('A.grad',A.grad)
print('b.grad',b.grad)

# Compare with dense op
A = Aref.detach().clone().to_dense().requires_grad_(True)
b = bref.detach().clone().requires_grad_(True)
x = torch.linalg.solve(A,b)
loss = x.sum()
loss.backward()
print('x',x)
print('allclose:',torch.allclose(A @ x, b))
print('A.grad',A.grad)
print('b.grad',b.grad)

# Now try some gradcheck
A = torch.randn(3,3, dtype=torch.float64, device=torchdevice).to_sparse()
A.requires_grad=True
b = torch.randn(3, dtype=torch.float64, requires_grad=True, device=torchdevice)
torch.autograd.gradcheck(sparsesolve, [A, b], check_sparse_nnz=True, raise_exception=True)
'''