#integration via BAOAB scheme

import numpy as np

#this is step A
def position_update(dim,n_particles,x,v,dt):    
    x_new=np.zeros((n_particles,dim))
    for i in range (0,n_particles):
        for j in range (0,dim):
            x_new[i][j] = x[i][j] + v[i][j]*dt/2.
    return x_new

#this is step B
def velocity_update(dim,n_particles,v,F,dt,m_atom):
    v_new=np.zeros((n_particles,dim))
    for i in range (0,n_particles):
        for j in range (0,dim):
            v_new[i][j] = v[i][j] + F[i][j]*dt/2./m_atom
    return v_new

#this is step O
def random_velocity_update(dim,n_particles,m_atom,v,gamma,kBT,dt):
    v_new=np.zeros((n_particles,dim))
    for i in range (0,n_particles):
        c1 = np.exp(-gamma*dt)
        c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT/m_atom)
        for j in range (0,dim):
            R = np.random.normal()
            v_new[i][j] = c1*v[i][j] + R*c2            
    return v_new

def baoab(dim,n_particles,potential, max_step_number, dt, gamma, kB,NA,temp,m_atom, initial_position, initial_velocity,k_vec,
                                     save_freq, **kwargs ):
    x = initial_position
    x0= initial_position
    v = initial_velocity
    kBT=kB*temp*NA
    save_frequency=save_freq
    t = 0
    step_number = 0
    max_time=max_step_number*dt
    max_time_ns=max_time/1e-09
    print ('Max simulation time = %f [ns]'%max_time_ns )
    max_storage=round(max_step_number/save_frequency)
    storage_cnt=0
    positions = np.zeros((max_storage,n_particles,dim))
    velocities = np.zeros((max_storage,n_particles,dim))
    forces = np.zeros((max_storage,n_particles,dim))
    temperature = np.zeros(max_storage)
    e_pot=np.zeros(max_storage)
    e_kin=np.zeros(max_storage)
    total_energies = np.zeros(max_storage)
    save_times = np.zeros(max_storage)
    while(step_number<max_step_number):
    #B
        potential_energy, force = potential(dim,n_particles,x,x0,k_vec)
        v = velocity_update(dim,n_particles,v,force,dt,m_atom)
        
    #A
        x = position_update(dim,n_particles,x,v,dt)

    #O
        v = random_velocity_update(dim,n_particles,m_atom,v,gamma,kBT,dt)
        
    #A
        x = position_update(dim,n_particles,x,v,dt)
    #B
        potential_energy, force = potential(dim,n_particles,x,x0,k_vec)
        v = velocity_update(dim,n_particles,v,force,dt,m_atom)
                   
        if step_number%save_frequency == 0 and step_number>=0:
            print(step_number)
            ekin=0
            for i in range (0,n_particles):
                ekin+=0.5*m_atom*np.dot(v[i],v[i].T)
                for j in range (0,dim):
                    positions[int(storage_cnt)][i][j]=x[i][j]
                    velocities[int(storage_cnt)][i][j]=v[i][j]
                    forces[int(storage_cnt)][i][j]=force[i][j]
            temperature[int(storage_cnt)]=2*ekin/n_particles/NA/kB/3   
            e_total =(ekin+ potential_energy)
            e_kin[int(storage_cnt)]=ekin
            e_pot[int(storage_cnt)]=potential_energy
            total_energies[int(storage_cnt)]=e_total
            save_times[int(storage_cnt)]=t
            storage_cnt+=1
        
        t = t+dt
        step_number = step_number + 1
    
    return (save_times, positions, velocities, total_energies, e_pot,temperature,forces)   

