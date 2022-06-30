#compute harmonic bonds between neighboring particles

import numpy as np

def harmonic_bond_force(dim,n_particles,x,x0,k):
    #calculate the energy on force on the right hand side of the equal signs
    energy=0
    force=np.zeros((n_particles,dim))
    for i in range (0,n_particles-1):
        j=i+1
        dx= x[i]-x[j]
        dist = np.linalg.norm(dx)
        if (i ==0)  or (i==2):
            k_val=k[0]
            dist0=10.0*1e-10 #[m]
        else:
            k_val=k[1]
            dist0=10.0*1e-10 #[m]
        energy += 0.5*k_val*(dist-dist0)**2
        #compute unit_vector to make force a vector
        u_vec=dx/dist
        force[i] =  force[i]-k_val*(dist-dist0)*u_vec
        force[j] =  force[j]+k_val*(dist-dist0)*u_vec
    return energy, force
