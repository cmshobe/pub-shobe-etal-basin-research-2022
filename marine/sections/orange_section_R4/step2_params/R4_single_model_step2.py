import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
import xsimlab as xs

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
    #br_vars = xs.group("br_vars") #allows for multiple processes influencing; say diffusion and subsidence

    z = xs.variable(
        dims="x", intent="inout", description="surface elevation z", attrs={"units": "m"}
    )
    #br = xs.variable(
    #    dims=[(), "x"], intent="in", description="bedrock_elevation", attrs={"units": "m"}
    #)
    br = xs.variable(
        dims=[(), "x"], intent="in", description="bedrock_elevation", attrs={"units": "m"}
    )
    h = xs.variable(
        dims="x", intent="inout", description="sed_thickness", attrs={"units": "m"}
    )
    
    dh_compact = xs.variable(
        dims="x", intent="inout", description="dh_compact", attrs={"units": "m/yr"}
    )

    def run_step(self):
        #self._delta_br = sum((br for br in self.br_vars))
        self._delta_h = sum((h for h in self.h_vars))

    def finalize_step(self):
        #self.br += self._delta_br #update bedrock surface
        self.h += self._delta_h #update sediment thickness
        self.z = self.br + self.h #add sediment to bedrock to get topo elev.

@xs.process
class ErosionDeposition:
    """Here's where the actual computation happens: nonlocal diffusion

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
    #dbr = xs.variable(dims="x", intent="out", groups="br_vars")
    dh = xs.variable(dims="x", intent="out", groups="h_vars")
    
    spacing = xs.foreign(UniformGrid1D, "spacing")
    #x = xs.foreign(UniformGrid1D, "x")
    z = xs.foreign(ProfileZ, "z")
    br = xs.foreign(ProfileZ, "br")
    h = xs.foreign(ProfileZ, "h")
    dh_compact = xs.foreign(ProfileZ, "dh_compact")
    
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
            self.deposition[first_marine_node] = qs_in / self.spacing #(self.qs_in * (1 + np.minimum(1, np.power(self.slope[first_marine_node] / self.s_crit, 2)))) / self.travel_dist #because slope = 0 #self.qs_in / self.travel_dist
        self.dh_dt[first_marine_node] = (-self.erosion[first_marine_node] + self.deposition[first_marine_node]) / (1 - self.sed_porosity)
        self.dh[first_marine_node] = self.dh_dt[first_marine_node] * dt
        self.qs[first_marine_node] = np.maximum(qs_in + (self.erosion[first_marine_node] - self.deposition[first_marine_node]) * self.spacing, 0.)
        if -self.dh[first_marine_node] > self.h[first_marine_node]:
            self.dh[first_marine_node] = -self.h[first_marine_node]
            self.qs[first_marine_node] = np.maximum(qs_in + ((-self.dh[first_marine_node] / dt) * (1 - self.sed_porosity)) * self.spacing, 0.)
        
        
        
        #evolve remaining marine nodes
        for i in range(first_marine_node + 1, len(self.z)): #iterate over each element of x ONLY IN THE MARINE
        
            if self.slope[i] <= 0: #this is the "regular," right-draining case
                self.deposition[i] = (self.qs[i-1] * (1 - np.minimum(1, np.power(self.slope[i] / self.s_crit, 2)))) / self.travel_dist
                self.erosion[i] = k_arr[i] * np.abs(self.slope[i])
                if self.z[i] > self.sea_level:
                    self.deposition[i] = 0
            else: #this is the irregular, left-draining case
                self.deposition[i] = self.qs[i-1] / self.spacing#(self.qs[i-1] * (1 + np.minimum(1, np.power(self.slope[i] / self.s_crit, 2)))) / self.spacing#self.travel_dist #self.qs[i-1] / self.spacing
                self.erosion[i] = 0
                if self.z[i] > self.sea_level:
                    self.deposition[i] = 0
            self.dh_dt[i] = (-self.erosion[i] + self.deposition[i]) / (1 - self.sed_porosity)
            
            self.dh[i] = self.dh_dt[i] * dt
            
            self.qs[i] = np.maximum(self.qs[i-1] + (self.erosion[i] - self.deposition[i]) * self.spacing, 0.)
            
            if -self.dh[i] > self.h[i]:
                self.dh[i] = -self.h[i]
                self.qs[i] = np.maximum(self.qs[i-1] + ((-self.dh[i] / dt) * (1 - self.sed_porosity)) * self.spacing, 0.)
            
        #calculate change in sed thickness
        #self.dh[:first_marine_node] = 0
        #self.dh[first_marine_node:] = self.dh_dt[first_marine_node:] * dt
        #self.dh[self.erosion > self.h] = -self.h[self.erosion > self.h] #if erosion is greater than h, topo only loses h    
        
        #compact sediment
        #calculate the thickness after compaction, z0; dh is the thickness of new deposited solid sediment
        self.dh_compact[:] = self.dh[:] * (1 - self.sed_porosity)
    
        
        #compaction routine
        #def compaction(porosity, porosity_depth_scale, nn, dh, zi):
        nn = len(self.z)
        z0 = np.zeros(nn)
        #set initial guess for z0:
        z0[:] = self.h[:]

        #Newton-Raphson iteration  at every node
        for k in range(nn):
            fx = 1 #throwaway initial value to trigger "while" loop
            dfx = 1 #throwaway initial value to trigger "while" loop
            #check whether we're at the root
            if self.dh_compact[k] > 0 : #only apply compaction where deposition is happening
                while np.abs(fx / dfx) > 1e-6:
                    #calculate value of function at initial guess
                    fx = z0[k] - self.h[k] + self.sed_porosity * self.sed_porosity_depth_scale * (np.exp(-z0[k] / self.sed_porosity_depth_scale) - np.exp(-self.h[k] / self.sed_porosity_depth_scale)) - self.dh_compact[k]
        
                    #calculate derivative
                    dfx = 1 - self.sed_porosity * np.exp(-z0[k] / self.sed_porosity_depth_scale)
                    z0[k] = z0[k] - fx / dfx

            elif self.dh_compact[k] == 0: #no e or d
                z0[k] = self.h[k]
            else: #in the case where erosion happens, the sediment surface shouldn't rebound. 
                z0[k] = self.h[k] + self.dh_compact[k] / (1 - self.sed_porosity)
        
        #here, have a chance to set the final dh by differencing new h (z0) and old h (h)
        self.dh[:] = z0[:] - self.h[:]
        
        #finalize changes to bedrock (subsidence) and sediment thickness (e/d)
        #self.dbr = (self.subsidence * dt)
        
@xs.process
class InitBasinGeom:
    """
    Give initial basin elevation field as a function of x:
    z = exp(- (x - a) / b) + c
    """
    
    #a = xs.variable(description="shift parameter", static=True)
    #b = xs.variable(description="scale parameter", static=True)
    #c = xs.variable(description="initial basin floor altitude", static=True)
    #d = xs.variable(description="x multiplier", static=True)

    init_br = xs.variable(dims="x", description="shift parameter", static=True)
    
    x = xs.foreign(UniformGrid1D, "x")
    z = xs.foreign(ProfileZ, "z", intent="out")
    #br = xs.foreign(ProfileZ, "br", intent="in")
    h = xs.foreign(ProfileZ, "h", intent="out")
    dh_compact = xs.foreign(ProfileZ, "dh_compact", intent="out")
    def initialize(self):
        #self.br = np.exp(- (self.x * self.d - self.a) / self.b) + self.c #build the initial topography
        self.h = np.zeros(len(self.x)) #initial sediment thickness is 0
        self.z = np.zeros(len(self.x)) + self.init_br #self.br#(np.exp(- (self.x * self.d - self.a) / self.b) + self.c) + self.h
        self.dh_compact = np.zeros(len(self.x))
        
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
#this is loading in the full m3/yr numbers directly from Baby et al 2019.

in_ds = xs.create_setup(
	model=marine,
	clocks={
		'time': np.arange(0, 130000000, 1000),
		'otime': np.append(np.arange(0, 130000000, 1000), 129999000)#np.append(np.arange(0, 130000000, 1000000), 129999000) #np.array([19999000])
	},
	master_clock='time',
	input_vars={
		'grid': {'length': 1380000., 'spacing': 10000.},
            'init': {'init_br': initial_bedrock},
            'erode': {
            	'k_factor': 0.1,
                'k_depth_scale': 100,
                's_crit': 0.05,
                'travel_dist': 200000,
                'sed_porosity': 0.56,
                'sed_porosity_depth_scale': 2830.,
                'sea_level': 0.,
                'qs_in': qs_array,
                'basin_width': 1.0
            },
		'profile': {'br': bedrock_elev_array},
   },
	output_vars={'profile__z': 'otime', 'profile__br': 'otime', 'profile__h': 'otime', 'profile__dh_compact': 'otime'}
)

with marine: 
    out_ds = in_ds.xsimlab.run(store="/scratch/cs00048/marine/sections/orange_section_R4/step2_params/best_fit_XX.zarr")
    
print('single model realization finished successfully')