##################################
#INVERSION SCRIPT FOR ORANGE BASIN SECTION R4
#RUNS THE NONLINEAR, NONLOCAL MODEL
#AND USES ONLY THE MODERN BATHYMETRIC SURFACE TO CALCULATE MISFIT

#Shobe, C.M., Braun, J., Yuan, X.P., Campforts, B., Gailleton, B., 
#Baby, G., Guillocheau, F., and Robin, C. (2022) Inverting passive
#margin stratigraphy for marine sediment transport dynamics over
#geologic time, Basin Research.

#Please cite the above paper when using this data or code.

#Model and inversion script written by Charlie Shobe (West Virginia University)
#Archived in 2022 in conjunction with resubmission of Shobe et al. (2022).
##################################

import pyabc
import numpy as np
import scipy.stats as st
import tempfile
import os
import pandas as pd
import xarray as xr
import xsimlab as xs

run_number = 0 #used to name results file
n_procs = 100 #number of realizations that can run at once; should match qsub request n_nosed * ppn
pop_size = 300 #number of "accepted" runs needed to move on to the next population. 
min_epsilon = 0.01 #misfit at which inversion will stop; set small to force script to do full set of populations
max_n_pops = 15 #max number of populations if epsilon isn't reached

#process parameter ranges
#note that convention in pyABC is (min, range) NOT (min, max).
#so true max is actually (min + max) as defined below

erode__travel_dist_min = 100000
erode__travel_dist_max = 200000

erode__k_factor_min = -3 #remember this one is logarithmic
erode__k_factor_max = 5 #remember this one is logarithmic

erode__k_depth_scale_min = 1
erode__k_depth_scale_max = 199

erode__s_crit_min = 0.01
erode__s_crit_max = 0.09

@xs.process
class UniformGrid1D:
    """Create 1D model grid with uniform spacing"""
    
    #grid parameters
    spacing = xs.variable(description="grid_spacing", static=True)
    length = xs.variable(description="grid total length", static=True)
    
    #x is an index variable, used for accessing the grid.
    x = xs.index(dims="x")
    
    #create the grid
    def initialize(self):
        self.x = np.arange(0, self.length, self.spacing)
        
@xs.process
class ProfileZ:
    """Compute the evolution of the elevation (z) profile"""
    
    h_vars = xs.group("h_vars") #allows for multiple processes influencing; say diffusion and subsidence

    z = xs.variable(
        dims="x", intent="inout", description="surface elevation z", attrs={"units": "m"}
    )

    br = xs.variable(
        dims=[(), "x"], intent="in", description="bedrock_elevation", attrs={"units": "m"}
    )
    h = xs.variable(
        dims="x", intent="inout", description="sed_thickness", attrs={"units": "m"}
    )

    def run_step(self):
        self._delta_h = sum((h for h in self.h_vars))

    def finalize_step(self):
        self.h += self._delta_h #update sediment thickness
        self.z = self.br + self.h #add sediment to bedrock to get topo elev.
        
import numba as nb

@nb.jit(nopython = True, cache = True)
def evolve_remaining_nodes(first_marine_node, z, slope, erosion, k_arr, s_crit, travel_dist, sea_level, deposition, qs, spacing, dh_dt, dh ,sed_porosity, dt,h):
    
    for i in range(first_marine_node + 1, len(z)): #iterate over each element of x ONLY IN THE MARINE
        
        if slope[i] <= 0: #this is the "regular," right-draining case
            deposition[i] = (qs[i-1] * (1 - np.minimum(1, np.power(slope[i] / s_crit, 2)))) / travel_dist
            erosion[i] = k_arr[i] * np.abs(slope[i])
            if z[i] > sea_level:
                deposition[i] = 0
        else: #this is the irregular, left-draining case
            deposition[i] = qs[i-1] / spacing
            erosion[i] = 0
            if z[i] > sea_level:
                deposition[i] = 0
        dh_dt[i] = (-erosion[i] + deposition[i]) / (1 - sed_porosity)

        dh[i] = dh_dt[i] * dt

        qs[i] = np.maximum(qs[i-1] + (erosion[i] - deposition[i]) * spacing, 0.)

        if -dh[i] > h[i]:
            dh[i] = -h[i]
            qs[i] = np.maximum(qs[i-1] + ((-dh[i] / dt) * (1 - sed_porosity)) * spacing, 0.)
            
            
@nb.jit(nopython = True, cache = True)      
def local_NewtonRaphson_scheme_for_compaction(nn,dh_compact,z0, h, sed_porosity, sed_porosity_depth_scale):
    #Newton-Raphson iteration  at every node
        for k in range(nn):
            fx = 1 #throwaway initial value to trigger "while" loop
            dfx = 1 #throwaway initial value to trigger "while" loop
            #check whether we're at the root
            if dh_compact[k] > 0 : #only apply compaction where deposition is happening
                while np.abs(fx / dfx) > 1e-6:
                    #calculate value of function at initial guess
                    fx = z0[k] - h[k] + sed_porosity * sed_porosity_depth_scale * (np.exp(-z0[k] / sed_porosity_depth_scale) - np.exp(-h[k] / sed_porosity_depth_scale)) - dh_compact[k]
        
                    #calculate derivative
                    dfx = 1 - sed_porosity * np.exp(-z0[k] / sed_porosity_depth_scale)
                    z0[k] = z0[k] - fx / dfx

            elif dh_compact[k] == 0: #no e or d
                z0[k] = h[k]
            else: #in the case where erosion happens, the sediment surface shouldn't rebound. 
                z0[k] = h[k] + dh_compact[k] / (1 - sed_porosity)

@xs.process
class ErosionDeposition:
    """Here's where the actual computation happenfirst_marine_nodecal diffusion

    """
    
    #get input parameters
    basin_width = xs.variable(description="basin width")
    sea_level = xs.variable(description="sea level")
    k_factor = xs.variable(description="k factor")
    k_depth_scale = xs.variable(description="k depth scale")
    travel_dist = xs.variable(description="marine_travel_distance")
    s_crit = xs.variable(description="marine_critical_slope")
    sed_porosity = xs.variable(description="marine_sediment_porosity")
    sed_porosity_depth_scale = xs.variable(description="marine_sediment_porosity_depth_scale")
    qs_in = xs.variable(description="sediment_flux_in")
    
    slope = xs.variable(
        dims="x", intent="out", description="topographic_slope", attrs={"units": "-"}
    )
    depth = xs.variable(
        dims="x", intent="out", description="depth", attrs={"units": "m"}
    )
    erosion = xs.variable(
        dims="x", intent="out", description="erosion", attrs={"units": "m/yr"}
    )
    deposition = xs.variable(
        dims="x", intent="out", description="deposition", attrs={"units": "m/yr"}
    )
    dh_dt = xs.variable(
        dims="x", intent="out", description="dh_dt", attrs={"units": "m/yr"}
    )
    qs = xs.variable(
        dims="x", intent="out", description="qs", attrs={"units": "m2/yr"}
    )
    dh = xs.variable(dims="x", intent="out", groups="h_vars")
    
    spacing = xs.foreign(UniformGrid1D, "spacing")
    z = xs.foreign(ProfileZ, "z")
    br = xs.foreign(ProfileZ, "br")
    h = xs.foreign(ProfileZ, "h")
    
    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        k_factor_real = np.power(10, self.k_factor)
        self.erosion = np.repeat(0., len(self.z))
        self.deposition = np.repeat(0., len(self.z))
        self.dh_dt = np.repeat(0., len(self.z))
        self.qs = np.repeat(0., len(self.z))
        self.dh = np.repeat(0., len(self.z))
        
        #divide Qs_in by basin width to get qs_in
        qs_in = self.qs_in / self.basin_width
        
        #calculate topographic slope
        self.slope = np.append(np.diff(self.z) / self.spacing, 0)
        #calculate depth below water
        self.depth = np.maximum(self.sea_level - self.z, 0)
        
        #calculate k array
        k_arr = k_factor_real * np.exp(-self.depth / self.k_depth_scale)
        
        #impose hard basin floor
        k_arr[self.z <= (self.br + 0.001)] = 0
        
        #find first marine node
        marine_or_terrestrial = self.z <= self.sea_level #boolean: true is marine
        first_marine_node = np.argmax(marine_or_terrestrial) #finds the first true
        
        #evolve first marine node
        if self.slope[first_marine_node] <= 0: #this is the "regular," right-draining case
            self.erosion[first_marine_node] = k_arr[first_marine_node] * np.abs(self.slope[first_marine_node])
            self.deposition[first_marine_node] = (qs_in * (1 - np.minimum(1, np.power(self.slope[first_marine_node] / self.s_crit, 2)))) / self.travel_dist

        else:  #this is the irregular, left-draining case
            self.erosion[first_marine_node] = 0 #because slope = 0
            self.deposition[first_marine_node] = qs_in / self.spacing
        self.dh_dt[first_marine_node] = (-self.erosion[first_marine_node] + self.deposition[first_marine_node]) / (1 - self.sed_porosity)
        self.dh[first_marine_node] = self.dh_dt[first_marine_node] * dt
        self.qs[first_marine_node] = np.maximum(qs_in + (self.erosion[first_marine_node] - self.deposition[first_marine_node]) * self.spacing, 0.)
        if -self.dh[first_marine_node] > self.h[first_marine_node]:
            self.dh[first_marine_node] = -self.h[first_marine_node]
            self.qs[first_marine_node] = np.maximum(qs_in + ((-self.dh[first_marine_node] / dt) * (1 - self.sed_porosity)) * self.spacing, 0.)
        
        
        
        #evolve remaining marine nodes
        evolve_remaining_nodes(first_marine_node, self.z, self.slope, self.erosion, k_arr, self.s_crit, 
                               self.travel_dist, self.sea_level, self.deposition, self.qs, self.spacing,
                               self.dh_dt, self.dh ,self.sed_porosity, dt, self.h)


        #compact sediment
        #calculate the thickness after compaction, z0; dh is the thickness of new deposited solid sediment
        dh_compact = self.dh * (1 - self.sed_porosity)
    
        
        #compaction routine
        #def compaction(porosity, porosity_depth_scale, nn, dh, zi):
        nn = len(self.z)
        z0 = np.zeros(nn)
        #set initial guess for z0:
        z0[:] = self.h[:]
        
        local_NewtonRaphson_scheme_for_compaction(nn,dh_compact,z0, self.h, self.sed_porosity, self.sed_porosity_depth_scale)

        #here, have a chance to set the final dh by differencing new h (z0) and old h (h)
        self.dh[:] = z0[:] - self.h[:]
        
@xs.process
class InitBasinGeom:
    """
    Set up initial basin geometry
    """

    init_br = xs.variable(dims="x", description="shift parameter", static=True)
    
    x = xs.foreign(UniformGrid1D, "x")
    z = xs.foreign(ProfileZ, "z", intent="out")
    h = xs.foreign(ProfileZ, "h", intent="out")

    def initialize(self):
        self.h = np.zeros(len(self.x)) #initial sediment thickness is 0
        self.z = np.zeros(len(self.x)) + self.init_br #self.br#(np.exp(- (self.x * self.d - self.a) / self.b) + self.c) + self.h

marine = xs.Model(
    {
        "grid": UniformGrid1D,
        "profile": ProfileZ,
        "init": InitBasinGeom,
        "erode": ErosionDeposition,
    }
)

#need to import basement elevation and qs time series after they were exported by the prepro notebook
bedrock_file = '/scratch/cs00048/marine/sections/orange_section_R4/prepro/bedrock_elev_array.npy'
br = np.load(bedrock_file)
bedrock_elev_array = xr.DataArray(br, dims=['time', 'x'])
initial_bedrock = bedrock_elev_array[0, :]

qs_file = '/scratch/cs00048/marine/sections/orange_section_R4/prepro/qs_array.npy'
qs_array = np.load(qs_file)
qs_array = xr.DataArray(qs_array, dims=['time'])

#set up model inputs
in_ds = xs.create_setup(
			model=marine,
			clocks={
				'time': np.arange(0, 130000000, 1000),
				'otime': np.array([129999000])
			},
			master_clock='time',
			input_vars={
				'grid': {'length': 1380000., 'spacing': 10000.},
                'init': {'init_br': initial_bedrock},
                'erode': {
                    'k_factor': 1.,
                    'k_depth_scale': 100.,
                    's_crit': 0.1,
                    'travel_dist': 10000.,
                    'sed_porosity': 0.56,
                    'sed_porosity_depth_scale': 2830.,
                    'sea_level': 0.,
                    'qs_in': qs_array,
                    'basin_width': 1.0,
                },
                'profile': {
                    'br': bedrock_elev_array
                },
   },
            output_vars={'profile__z': 'otime', 'profile__br': 'otime', 'profile__h': 'otime'}
)

#set up an instance of the model to feed to pyABC
def model(parameter):
    model = marine.clone()
    with model: 
        #write param values out
        travel_dist = parameter['erode__travel_dist']
        k_factor = parameter['erode__k_factor']
        k_depth = parameter['erode__k_depth_scale']
        s_crit = parameter['erode__s_crit']
        ds_out = (
            in_ds
            .xsimlab.update_vars(input_vars=parameter)
            .xsimlab.run()
        )
    return {"data": ds_out.profile__h[-1, 1:], "params_for_record":[travel_dist, k_factor, k_depth, s_crit]}
    
prior = pyabc.Distribution(erode__travel_dist=pyabc.RV("uniform", erode__travel_dist_min, erode__travel_dist_max),
                           erode__k_factor=pyabc.RV("uniform", erode__k_factor_min, erode__k_factor_max),
                           erode__k_depth_scale=pyabc.RV("uniform", erode__k_depth_scale_min, erode__k_depth_scale_max),
                           erode__s_crit=pyabc.RV("uniform", erode__s_crit_min, erode__s_crit_max))

#calculate misfit                                                                              
def distance(x, y): #x is simulated, y is observed
    misfit = np.sqrt((1/137) * np.sum(np.power(y["data"] - x["data"], 2)/np.power(10,2)))
        
    #write params
    params_list = x["params_for_record"]
    string = str(params_list[0]) + ',' + str(params_list[1]) + ',' + str(params_list[2]) + ',' + str(params_list[3]) + ',' + str(np.array(misfit)) + '\n'
    with open('/scratch/cs00048/marine/sections/orange_section_R4/step2_params/all_params_' + str(run_number).zfill(3) + '.csv','a') as file:
        file.write(string) 

    return np.double(misfit)

#set up pyABC ABC-SMC algorithm  
sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs = n_procs)
abc = pyabc.ABCSMC(model, prior, distance, sampler = sampler,
                   population_size = pop_size)

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))

#define observation data to match
sed_surfaces = np.load('/scratch/cs00048/marine/sections/orange_section_R4/prepro/all_surfaces.npy')
top_surface = sed_surfaces[-1, :] #this is the top of the U8 surface
br_surface = sed_surfaces[0, :]
data_full_h = top_surface[1:] - br_surface[1:]
observation = data_full_h
abc.new(db_path, {"data": observation})

#run pyABC ABC-SMC algorithm
history = abc.run(minimum_epsilon=min_epsilon, max_nr_populations=max_n_pops)

#save history dataframe as csv
return_df = history.get_population_extended(t = "all")
return_df.to_csv('/scratch/cs00048/marine/sections/orange_section_R4/step2_params/step2_results_' + str(run_number).zfill(3) + '.csv')

#save best fit parameters so that I can do a single run later on
sorted_by_fit = return_df.sort_values('distance')
particle_id_of_best_fit = sorted_by_fit.iloc[0, 4]
best_fit_params = sorted_by_fit[sorted_by_fit['particle_id'] == particle_id_of_best_fit]
best_fit_travel_dist = best_fit_params['par_val'][best_fit_params['par_name'] == 'erode__travel_dist']
best_fit_k = best_fit_params['par_val'][best_fit_params['par_name'] == 'erode__k_factor']
best_fit_depth_scale = best_fit_params['par_val'][best_fit_params['par_name'] == 'erode__k_depth_scale']
best_fit_s_crit = best_fit_params['par_val'][best_fit_params['par_name'] == 'erode__s_crit']
best_fit_params = np.array([np.float(best_fit_travel_dist.iloc[0]), 
                           np.float(best_fit_k.iloc[0]), 
                           np.float(best_fit_depth_scale.iloc[0]), 
                           np.float(best_fit_s_crit.iloc[0])]
                          )
np.save('/scratch/cs00048/marine/sections/orange_section_R4/step2_params/best_fit_params_' + str(run_number).zfill(3), best_fit_params)
print('inversion script complete.')
