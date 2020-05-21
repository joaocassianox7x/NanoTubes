#FITA E TUBO

#O PROGRAMA E FORTENMENTE BASEADO NO PROGRAMA DO GEORGE, INFELIZMENTE MINHAS 
#PRIMEIRAS TENTATIVAS FRACASSARAM MISERAVELMENTE

#PACOTES
import numpy as np
import numpy.linalg as alg
import time
import multiprocessing as mp
from constantes import *
import os

def gen_matriz(Ef=0):
    t1j1=np.zeros((3,nsite,norbit,nsite,norbit),dtype=np.longcomplex) #primeiros hoppings
    t1j2=np.zeros((3,nsite,norbit,nsite,norbit),dtype=np.longcomplex) #segundos hoppings
    #-------------------
    
    #DEFINICAO DO HAMILTONIANO

    for i1 in range(nsite): #primeiro for/DO
        for i2 in range(norbit-2):
            t1j1[1,i1,i2,i1,i2+2]=v1 #primeiro hopping
            t1j1[1,i1,i2+2,i1,i2]=v1 #primeiro hopping
       
        t1j1[2,i1,0,i1,2]=v1
        t1j1[0,i1,2,i1,0]=v1
    
        t1j1[2,i1,1,i1,3]=v1
        t1j1[0,i1,3,i1,1]=v1
        
        t1j1[0,i1,4,i1,6]=v1
        t1j1[2,i1,6,i1,4]=v1
        
        t1j1[0,i1,5,i1,7]=v1
        t1j1[2,i1,7,i1,5]=v1
    
    for i1 in range(nsite-1): #segundo e terceiro DO/for
        i2=i1+1
        
        t1j1[1,i1,6,i2,0]=v1
        t1j1[1,i1,7,i2,1]=v1
        
        t1j1[1,i2,0,i1,6]=v1
        t1j1[1,i2,1,i1,7]=v1
    
    for i1 in range(nsite):
        t1j2[2,i1,0,i1,0]=lso*lsofact*-1*1j
        t1j2[0,i1,0,i1,0]=lso*lsofact*1*1j
        t1j2[2,i1,1,i1,1]=lso*lsofact*1*1j
        t1j2[0,i1,1,i1,1]=lso*lsofact*-1*1j
        
        t1j2[2,i1,2,i1,2]=lso*lsofact*1*1j
        t1j2[0,i1,2,i1,2]=lso*lsofact*-1*1j
        t1j2[2,i1,3,i1,3]=lso*lsofact*-1*1j
        t1j2[0,i1,3,i1,3]=lso*lsofact*1*1j
        
        t1j2[2,i1,4,i1,4]=lso*lsofact*-1*1j
        t1j2[0,i1,4,i1,4]=lso*lsofact*1*1j
        t1j2[2,i1,5,i1,5]=lso*lsofact*1*1j
        t1j2[0,i1,5,i1,5]=lso*lsofact*-1*1j
        
        t1j2[2,i1,6,i1,6]=lso*lsofact*1*1j
        t1j2[0,i1,6,i1,6]=lso*lsofact*-1*1j
        t1j2[2,i1,7,i1,7]=lso*lsofact*-1*1j
        t1j2[0,i1,7,i1,7]=lso*lsofact*1*1j
    
    #hopping intra-minor cell
        
    #diagonal superior
        t1j2[1,i1,0,i1,4]=lso*lsofact*(-1)*1j
        t1j2[1,i1,1,i1,5]=lso*lsofact*(1)*1j
        t1j2[1,i1,2,i1,6]=lso*lsofact*(-1)*1j
        t1j2[1,i1,3,i1,7]=lso*lsofact*(1)*1j
    
    #diagonal inferior
        t1j2[1,i1,4,i1,0]=lso*lsofact*(1)*1j
        t1j2[1,i1,5,i1,1]=lso*lsofact*(-1)*1j
        t1j2[1,i1,6,i1,2]=lso*lsofact*(1)*1j
        t1j2[1,i1,7,i1,3]=lso*lsofact*(-1)*1j
    
    #hopping inter-minor celL
        
    #diagonal superior
        t1j2[2,i1,0,i1,4]=lso*lsofact*(1)*1j
        t1j2[2,i1,1,i1,5]=lso*lsofact*(-1)*1j
        t1j2[0,i1,2,i1,6]=lso*lsofact*(1)*1j
        t1j2[0,i1,3,i1,7]=lso*lsofact*(-1)*1j
    
    #diagonal inferor
        t1j2[0,i1,4,i1,0]=lso*lsofact*(-1)*1j
        t1j2[0,i1,5,i1,1]=lso*lsofact*(1)*1j
        t1j2[2,i1,6,i1,2]=lso*lsofact*(-1)*1j
        t1j2[2,i1,7,i1,3]=lso*lsofact*(1)*1j
    
    for i1 in range(nsite-1):
        i2=i1+1
        
        #independente de k
        t1j2[1,i1,4,i2,0]=lso*lsofact*1j*(1)
        t1j2[1,i1,5,i2,1]=lso*lsofact*1j*(-1)
        t1j2[1,i1,6,i2,2]=lso*lsofact*1j*(1)
        t1j2[1,i1,7,i2,3]=lso*lsofact*1j*(-1)
    
        #dependente de k
        t1j2[0,i1,4,i2,0]=lso*lsofact*1j*(-1)
        t1j2[0,i1,5,i2,1]=lso*lsofact*1j*(1)
        t1j2[2,i1,6,i2,2]=lso*lsofact*1j*(-1)
        t1j2[2,i1,7,i2,3]=lso*lsofact*1j*(1)
    
        #herminttiano conjugado da dependente de k
        t1j2[1,i2,0,i1,4]=lso*lsofact*1j*(-1)
        t1j2[1,i2,1,i1,5]=lso*lsofact*1j*(1)
        t1j2[1,i2,2,i1,6]=lso*lsofact*1j*(-1)
        t1j2[1,i2,3,i1,7]=lso*lsofact*1j*(1)
    
        #hermitiano conjugado da independente
        t1j2[2,i2,0,i1,4]=lso*lsofact*1j*(1)
        t1j2[2,i2,1,i1,5]=lso*lsofact*1j*(-1)
        t1j2[0,i2,2,i1,6]=lso*lsofact*1j*(1)
        t1j2[0,i2,3,i1,7]=lso*lsofact*1j*(-1)
    
        #edicoes para o tubo####################
    m=0
    m2=0
    ra=[Ra,Ra,Rb,Rb]*2
    
    for i1 in range(nsite):
        for i2 in range(norbit):
            m+=1
            m2+=1
            if m2==2:
                theta=((m-1)*np.pi*2)/(nsite*norbit)
                t1j1[1,i1,i2,i1,i2]=(-Ef/(2*ll))*(np.cos(theta)*ra[i2]+Ra)
                t1j1[1,i1,i2-1,i1,i2-1]=(-Ef/(2*ll))*(np.cos(theta)*ra[i2]+Ra)
                m2=0
                
    t1j1[1,0,0,nsite-1,6]=v1
    t1j1[1,0,1,nsite-1,7]=v1
    t1j1[1,nsite-1,6,0,0]=v1
    t1j1[1,nsite-1,7,0,1]=v1
    
    t1j2[1,0,0,nsite-1,4]=lso*lsofact*-1*1j
    t1j2[1,0,1,nsite-1,5]=lso*lsofact*-1*-1j
    t1j2[1,0,2,nsite-1,6]=lso*lsofact*-1*1j
    t1j2[1,0,3,nsite-1,7]=lso*lsofact*-1*-1j
        
    t1j2[1,nsite-1,0,0,4]=lso*lsofact*-1*1j
    t1j2[1,nsite-1,1,0,5]=lso*lsofact*-1*-1j
    t1j2[1,nsite-1,2,0,6]=lso*lsofact*-1*1j
    t1j2[1,nsite-1,3,0,7]=lso*lsofact*-1*-1j
    
    t1j2[2,0,0,nsite-1,4]=lso*lsofact*1j*1
    t1j2[2,0,1,nsite-1,5]=lso*lsofact*1j*-1
    t1j2[0,0,2,nsite-1,6]=lso*lsofact*1j*1
    t1j2[0,0,3,nsite-1,7]=lso*lsofact*1j*-1
        
    t1j2[0,nsite-1,0,0,4]=lso*lsofact*1j*-1
    t1j2[0,nsite-1,1,0,5]=lso*lsofact*1j*1
    t1j2[2,nsite-1,2,0,6]=lso*lsofact*1j*-1
    t1j2[2,nsite-1,3,0,7]=lso*lsofact*1j*1
    
    return(t1j1,t1j2)

def tight(Ef=0):
    t1j1,t1j2=gen_matriz(Ef)
    eigenval=np.zeros((2*ngridy+2,nsite,norbit))
    hamil=np.zeros((nsite*norbit,nsite*norbit),dtype=complex)
    coeficientes=np.zeros((2*ngridy,nsite,norbit,nsite,norbit))
    sinal=[-1,0,1]
    K=np.linspace(0,np.pi,2*ngridy)
    l=0
    for k in K:
        for i in range(3):
            fase=k*sinal[i]
            for is1 in range(nsite):
                for is2 in range(nsite):
                    for k2 in range(norbit):
                        for k3 in range(norbit):
                            j1=norbit*(is1-1)+k2
                            j2=norbit*(is2-1)+k3
                            hamil[j1,j2]+=(t1j2[i,is1,k2,is2,k3]-t1j1[i,is1,k2,is2,k3])*(np.cos(fase)-1j*np.sin(fase))
                            if j1==j2:
                                hamil[j1,j2]=0
        a,b=alg.eigh(hamil)
        p=0
        for is1 in range(nsite): 
            for k1 in range(norbit):
                eigenval[l,is1,k1]=a[p]
                p+=1

        for is1 in range(nsite):
            for k1 in range(norbit):
                j1=norbit*is1+k1
                for is2 in range(nsite):
                    for k2 in range(norbit):
                        j2=norbit*is2+k2
                        coeficientes[l,is2,k2,is1,k1]=np.abs(b[j2,j1])**2
        l+=1
        hamil=np.zeros((nsite*norbit,nsite*norbit),complex)
    
    bandas=np.zeros((len(K),nsite*norbit+1))
    bandas[:,0]=k
    for i1 in range(len(K)):    
        for i2 in range(nsite):
            for i3 in range(norbit):
                bandas[i1,i2+i3+1]=eigenval[i1,i2,i3]
    
    arq=open('Bandas/'+'E_'+str(round(Ef,4))+'.txt','w')
    for i1 in range(len(K)):
        arq.write(str(round(K[i1],cp)))
        arq.write(' ')
        for i2 in range(nsite):
            for i3 in range(norbit):
                arq.write(str(round(eigenval[i1,i2,i3],cp)))
                arq.write(' ')
        arq.write('\n')
    
    
    arq.close()
    
    return eigenval,coeficientes,K


def green(Ef=0):
    eigenval,coeficientes,K=tight(Ef)
    omega=np.zeros((energrid))
    nsites=nsite
    norbits=norbit
    ldos=np.zeros((energrid,nsites,norbits))
    vec_ener=np.linspace(enerinit,-enerinit,energrid)
    for i1 in range(energrid):
        omega[i1]=enerinit+bandwidth*i1/energrid
        
    for i1 in range(0,2*ngridy):
        for i3 in range(nsites):
            for i4 in range(norbits):
                for i2 in range(energrid):
                    green=1/(omega[i2]-eigenval[i1,i3,i4]+eta)
                    for i5 in range(nsites):
                        for i6 in range(norbits):
                            ldos[i2,i5,i6]=ldos[i2,i5,i6]-coeficientes[i1,i5,i6,i3,i4]*np.imag(green)/np.pi/(2*ngridy)
    green_matrix=np.zeros((energrid,nsite*norbit+1))
    green_matrix[:,0]=vec_ener[:]

    for i1 in range(energrid):
        l1=1
        for i2 in range(nsites):
            for i3 in range(norbits):
                green_matrix[i1,l1]=ldos[i1,i2,i3]
                l1+=1
    np.savetxt('Green/green'+str(round(Ef,4))+'.txt',green_matrix)
    

    nome='E_'+str(round(Ef,4))
    k=np.loadtxt('Bandas/'+nome+'.txt')
    x=k[:,0]
    sitio=nsite*4*2 #numero de sitios
    meio=sitio/2 #constante

    maxi=max(k[:,int(meio)])
    mini=min(k[:,int(meio)+1])
    
    band_sup=k[:,int(meio)+1]
    band_inf=k[:,int(meio)]
    
    i,=np.where(band_inf==maxi) #indice onde o maximo acontece na banda inferior
    j,=np.where(band_sup==mini) #"" na banda superior
    
    k=x[i][0]
    arq=open('x_gap.txt','a')
    arq.write(str(k)+'   '+str(Ef)+'\n')
    arq.close()

def estados(Ef=0):
    #Para os estados
    t1j1,t1j2=gen_matriz(Ef)
    sinal=[-1,0,1]
    hamil=np.zeros((nsite*norbit,nsite*norbit),complex)
    k=np.float64(-2.09439510234375)
    for i in range(3):
        fase=k*sinal[i]
        for is1 in range(nsite):
            for is2 in range(nsite):
                for k2 in range(norbit):
                    for k3 in range(norbit):
                        j1=norbit*(is1-1)+k2
                        j2=norbit*(is2-1)+k3
                        hamil[j1,j2]+=(t1j2[i,is1,k2,is2,k3]-t1j1[i,is1,k2,is2,k3])*(np.cos(fase)-1j*np.sin(fase))
    a,b=alg.eigh(hamil)
    b=np.abs(b)**2
    c=np.zeros((len(b[:,0]),4))
    c[:,0]=b[:,nsite*4-2]
    c[:,1]=b[:,nsite*4-1]
    c[:,2]=b[:,nsite*4]
    c[:,3]=b[:,nsite*4+1]
    np.savetxt("Estados/b"+str(Ef)+".txt",c)
 
def write_cons(A=True):
    if A==True:
        arq=open('constantes.txt','w')
        arq.write(' ngridy= '+str(ngridy))
        arq.write('\n nsite= '+str(nsite))
        arq.write('\n norbit= '+str(norbit))
        arq.write('\n casas de precisao= '+str(cp))
        arq.write('\n lso= '+str(lso))
        arq.write('\n lsofact= '+str(lsofact))
        arq.write('\n V1= '+str(v1))
        arq.write('\n Ei= '+str(Einicial)+' ,Ef= '+str(Efinal)+' ,Pts= '+str(len(campos)))
        arq.write('\n Energrid= '+str(energrid))
        arq.write('\n BandWidth= '+str(bandwidth))
        arq.write('\n Enerinit= '+str(enerinit))
        arq.write('\n Eta= '+str(eta))
        arq.close()
    
    else:
        return

###-----------------------###
def clr_old(A=False):
    if A==True:
        os.system('./clearall.sh')
    else:
        return

clr_old(True) #Apagar resultados anteriores?
###-----------------------###

def three(Ef): #Precisam ser inicializados nessa ordem
    write_cons(False)
    #tight(Ef) #apeas o tight
    green(Ef) #roda o tight e o ldos
    estados(Ef) #pode ser rodada de forma isolada

if __name__=='__main__':
    pool=mp.Pool(processes=len(campos)/4)
    pool.map(three, campos)

