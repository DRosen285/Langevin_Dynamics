#Run Langevin Dynamics (inspired by https://hockygroup.hosting.nyu.edu/exercise/langevin-dynamics.html)

import numpy as np
import Maxwell_Boltzmann_init
import Langevin_integrator
import force_field

#natural constants
kB=1.380649e-26 #[kJ/K]
NA=6.02214199e23 #[1/mol]

#md input
max_step_number=5e07
temp=2.045 #[K]
kBT=kB*temp*NA
#dimension of the sytem: 3=x,y,z; 2=x,y; 1=x
dim=3
delta_temp= 1*1e-12 #[s]: 1e-12 [s] cooresponds to [ps]
#friction coefficient for Langevin thermostat
my_gamma=0.5/delta_temp 
#time step for integrating Newton's equations of motion [s]: 1e-15 [s] cooresponds to [fs]
my_dt=1*1e-15 
#frequency to store energy,forces,positions:
save_freq=1000

#sytsem specific: 4 beads connected by a harmonic spring 

n_particles = 4
start_pos = np.zeros((n_particles,3))
spacing=1.0
start_pos[0,0] = 0
start_pos[1,0] = 1.0+(0.01*np.random.randn())
start_pos[2,0] = 2*1.0
start_pos[3,0] = 3*1.0
start_pos[:,1] = 0
start_pos[:,2] = 0
initial_position =start_pos*1e-09 #nm in m

m_atom=12/1000 #[kg/mol]
masses=np.zeros(n_particles)
for i in range (0,n_particles):
    masses[i]=m_atom

m_total=n_particles*m_atom

#generate initial velocities
initial_velocity= Maxwell_Boltzmann_init.init_velo(n_particles,masses,temp,kBT,kB,NA) # in [m/s]

#force field setup

#spring constant for harmonic bond
spring_const=50*temp
k_vec=[10*kB*NA*spring_const/(1e-09*1e-09),1.0*kB*NA*spring_const/(1e-09*1e-09)] #[kj/mol/m^2] --->1e-10: Angstrom; --->1e-09:nm
ff=force_field.harmonic_bond_force#force field

#run Langevin Dynamics
times, positions, velocities, total_energies,e_pot,temperature,forces = Langevin_integrator.baoab(dim,n_particles,ff, \
                                                                            max_step_number, my_dt, my_gamma, kB,NA,temp,m_atom,\
                                                                            initial_position, initial_velocity,\
                                                                            k_vec,save_freq)                  
