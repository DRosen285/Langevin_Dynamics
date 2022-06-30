import numpy as np 

#compute initial particle momenta based on Maxwell-Boltzmann distribution
def init_velo(n_particles,masses,temp,kBT,kB,NA):
        eps_temp = 1e-12 #avoid division by 0: temp_0=0 K
        vi = np.random.standard_normal((n_particles, 3))
        momenta = vi * np.sqrt(masses * kBT)[:, np.newaxis]
        vel=np.zeros((n_particles,3))
        ekin=0
        for i in range(0,n_particles):
            for j in range(0,3):
                vel[i][j]=momenta[i][j]/masses[i]
            ekin+=0.5*masses[i]*np.dot(vel[i],vel[i])
        temp_0 = 2*ekin/n_particles/NA/kB/3
    #temp_0 = 2*ekin/n_particles/kB/3
        temperature=temp
    #momenta are rescaled so the kinetic energy is  exactly 3/2 N k T.  
    #slight deviation from the correct Maxwell-Boltzmann distribution.
        if temperature > eps_temp:
            gamma = temperature / temp_0
        else:
            gamma = 0.0  
        momenta=momenta * np.sqrt(gamma)    
        for i in range(0,n_particles):
            for j in range(0,3):
                vel[i][j]=momenta[i][j]/masses[i]    
        return vel
