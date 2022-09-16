import torch as th
import math,random

def rasterize(pos,rad,radext,szCell,nrCell,reg=0.):
    '''
    I:      B x nx x ny
    pos:    B x np x 2, paricle locations
    rad:    radius of agent
    range:  softness range
    szCell: size of cell
    reg:    reg=0 has no effect, reg>0 implies normalized by I/(I+reg)
    '''
    invSzCell=1/szCell
    dummy=th.zeros((pos.shape[1],))
    rangeCell=math.ceil((rad+radext)*invSzCell)
    invDistHalf=2./(rad+radext)
    
    I=th.zeros((pos.shape[0],           \
                nrCell[0]+rangeCell*2+1,\
                nrCell[1]+rangeCell*2+1))
    for i in range(pos.shape[0]):
        #computer center
        x=(pos[i,:,0]*invSzCell).floor()
        y=(pos[i,:,1]*invSzCell).floor()
        #compute range
        for dx in range(-rangeCell,rangeCell+2):
            for dy in range(-rangeCell,rangeCell+2):
                #distance/kernel
                xoff=(x+dx)*szCell-pos[i,:,0]
                yoff=(y+dy)*szCell-pos[i,:,1]
                dist=(xoff*xoff+yoff*yoff+1e-8).sqrt()*invDistHalf
                Wq=(1-dist/2)**3*(1.5*dist+1)   #wendland quintic kernel
                #assign
                dxid=x+dx+rangeCell 
                dyid=y+dy+rangeCell
                I[i,dxid.long(),dyid.long()]+=th.where(dist<2,Wq,dummy)
    ICtr=I[:,rangeCell:I.shape[1]-rangeCell,rangeCell:I.shape[2]-rangeCell]
    if reg>0.:
        return ICtr/(ICtr+reg)
    else: return ICtr
    
if __name__=='__main__':
    #we need to do a finite difference check, which requires high-prec float64
    th.set_default_dtype(th.float64)
    
    #setup 
    bs=3
    N=10
    reg=.1
    rad=.5
    radext=.5
    szCell=.1
    nrCell=[200,100]
    pos=th.zeros((bs,N,2))
    for b in range(bs):
        for i in range(N):
            pos[b,i,0]=random.uniform(rad,szCell*nrCell[0]-rad)
            pos[b,i,1]=random.uniform(rad,szCell*nrCell[1]-rad)
    
    #forward/backward, finite difference check
    pos.requires_grad_(True)
    I=rasterize(pos,rad,radext,szCell,nrCell,reg)
    coef=th.rand((bs,nrCell[0]+1,nrCell[1]+1))
    loss=th.tensordot(I,coef,3)
    #get grad
    loss.backward()
    grad=pos.grad
    #compute loss
    delta=1e-8
    pos2=th.rand((bs,N,2))
    I2=rasterize(pos+pos2*delta,rad,radext,szCell,nrCell,reg)
    loss2=th.tensordot(I2,coef,3)
    analytic=th.tensordot(grad,pos2,3).item()
    numeric=(loss2.item()-loss.item())/delta
    print('loss=%f, grad=%f, error=%f!'%(loss.item(),analytic,analytic-numeric))
    
    #show image
    import matplotlib.pyplot as plt
    plt.imshow(I[0,:].T.detach().numpy())
    plt.show()