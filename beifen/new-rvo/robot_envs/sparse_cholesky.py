from cupyx.scipy.sparse.linalg import SuperLU

class SparseCholesky(SuperLU):
    def __init__(self, obj):
        from sksparse.cholmod import Factor
        if not isinstance(obj, Factor):
            raise TypeError('obj must be sksparse.cholmod.Factor')
        
        import cupy
        from cupyx.scipy import sparse
        self.shape = (len(obj.P()), len(obj.P()))
        self.nnz = obj.L().nnz
        self.perm_r = self.perm_c = cupy.argsort(cupy.array(obj.P()))
        self.L = sparse.csr_matrix(obj.L().tocsr())
        self.U = sparse.csr_matrix(obj.L().transpose().tocsr())
        self._perm_r_rev = self._perm_c_rev = obj.P()

def factorize(A):
    from sksparse.cholmod import cholesky
    fact = cholesky(A)
    return SparseCholesky(fact)

if __name__ == '__main__':
    
    def build_laplace(N, reg):
        from scipy.sparse import csc_matrix
        vals, row, col = [], [], []
        def id(r,c):
            return r+c*N
        def addDiagEntry(r,c,val):
            vals.append(val)
            row.append(id(r,c))
            col.append(id(r,c))
        def addEntry(r,c,r2,c2,val):
            if r2<0 or r2>=N:
                return
            if c2<0 or c2>=N:
                return
            vals.append(val)
            row.append(id(r,c))
            col.append(id(r,c))
            vals.append(-val)
            row.append(id(r,c))
            col.append(id(r2,c2))
        for r in range(N):
            for c in range(N):
                addDiagEntry(r,c,reg)
                for ro,co in [(1,0),(-1,0),(0,1),(0,-1)]:
                    addEntry(r,c,r+ro,c+co,1)
        return csc_matrix((vals, (row, col)), shape=[N*N, N*N])
    
    def build_laplace_rhs(N):
        import cupy
        return cupy.random.rand(N*N)

    A = build_laplace(200, 0.01)
    b = build_laplace_rhs(200)
    print("b=",b)
    
    chol = factorize(A)
    x = chol.solve(b)
    print("Ax=",A.dot(x.get()))
    print("Ax-b=",A.dot(x.get())-b.get())