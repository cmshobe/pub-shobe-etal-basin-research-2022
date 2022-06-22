import numpy as np
import xarray as xr

#########################################
#PREPROCESSING SCRIPT FOR ORANGE BASIN SECTION R5

#This script imports and prepares stratigraphic data from the following publication:

#Baby, G., Guillocheau, F., Braun, J., Robin, C., and Dall’Asta, M., 2019, Solid 
#sedimentation rates history of the Southern African continental margins: 
#Implications for the uplift history of the South African Plateau, Terra Nova, 
#v. 32, no. 1, p. 53—65, doi:10.1111/ter.12435.

#for use in the model-data comparison exercise detailed in:

#Shobe, C.M., Braun, J., Yuan, X.P., Campforts, B., Gailleton, B., 
#Baby, G., Guillocheau, F., and Robin, C. (2022) Inverting passive
#margin stratigraphy for marine sediment transport dynamics over
#geologic time, Basin Research.

#Please cite the above papers when using this data or code.

#Preprocessing code written by Charlie Shobe (West Virginia University)
#Archived in 2022 in conjunction with resubmission of Shobe et al. (2022).

#Please note that stratigraphic line drawings were prepared in Adobe Illustrator
#to allow removal of some complicating features (e.g., recent seamounts). This 
#script and the accompanying line drawings are not the dataset of record for Southeast
#Atlantic Margin stratigraphy. Rather, it is the dataset of record for the model-data
#comparison exercise in Shobe et al. (2022).

#########################################

#Define model domain
n_nodes = 77
dx = 10000
x = np.arange(0, n_nodes*dx, dx)
br_elev_array = np.zeros((130000, n_nodes))

#Extract arrays giving sediment surface elevations from line drawings
from xml.dom import minidom

def extract_surface(filename):

    doc = minidom.parse(filename)  # parseString also exists
    path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
    doc.unlink()
    top_of_syn_rift = path_strings[-1] #ignore all the other paths, which are the grid lines
    top_of_syn_rift_no_m = top_of_syn_rift[1:]
    n_coords = top_of_syn_rift.count(',')
    x_coords = np.zeros(n_coords)
    y_coords = np.zeros(n_coords)
    coords_list = top_of_syn_rift_no_m.split('L')
    iter = 0
    for i in coords_list:
        a = i.split(',')
        x_coords[iter] = float(a[0])
        y_coords[iter] = float(a[1])
        iter += 1
    
    x_coords = x_coords
    return x_coords, y_coords
    
#this isn't yet scaled to the actual height. To do this:
    #for x values: (x / x_max) * length pf section (km)
    #for y values: (y/y_max) * max real elevation (km)

def scale_surface(x_imported, y_imported):
    #scaling needs to be very different in R5 because 
    #not all lines are same length (x)
    section_max_x = 774979 #774.4 km; true length of the section for scaling
    #the reason max_drawn_x is needed in this seciton si that not all drawn lines are drawn to the end
    #of the section.
    max_drawn_x = 924.983 #this needs to be the largest x (pre-scaling) of any section line
    y_imported = -y_imported #[::-1]
    x_new = -(x_imported / max_drawn_x) * section_max_x + section_max_x
    y_new = y_imported + abs(min(y_imported))
    #46.582 is the largest y value of any section line
    y_new = (y_new / 46.582) * 6545 #rescale sections using measurements from vector editor
    y_new -= max(y_new)

    return x_new, y_new


from scipy.signal import savgol_filter
def smooth_surface(y_new, window_size):

    y_smoothed = savgol_filter(y_new, window_size, 3)
    return y_smoothed
    
def filter_surface(x_new, y_smoothed, n_x_nodes):
    test_x_new_smoothed = x_new[:len(y_smoothed)]
    final_cond = np.zeros(n_x_nodes)
    for j in range(len(x)):
        index_of_nearest_x = (np.abs(test_x_new_smoothed-x[j])).argmin()
        final_cond[j] = y_smoothed[index_of_nearest_x]
    return final_cond

#list names of svgs from which to import
folder = '/users/cs00048/marine/sections/orange_section_R5/prepro/strat_data_lines/'
filenames = [folder + '4_A20_RC1703_edit_TD_CS_top_of_synrift_only.svg', 
             folder + '4_A20_RC1703_edit_TD_CS_top_of_U1_only.svg', 
             folder + '4_A20_RC1703_edit_TD_CS_top_of_U2_only.svg', 
             folder + '4_A20_RC1703_edit_TD_CS_top_of_U3_only.svg', 
             folder + '4_A20_RC1703_edit_TD_CS_top_of_U4_only.svg', 
             folder + '4_A20_RC1703_edit_TD_CS_top_of_U5_only.svg', 
             folder + '4_A20_RC1703_edit_TD_CS_top_of_U6_only.svg', 
             folder + '4_A20_RC1703_edit_TD_CS_top_of_U7_only.svg', 
             folder + '4_A20_RC1703_edit_TD_CS_top_of_U8_only.svg', ]


def import_and_transform_all_surfaces(filenames):
    array_of_surfaces = np.zeros((len(filenames), n_nodes))
    i = 0
    for name in filenames:
        #import surface
        x_imported, y_imported = extract_surface(name)
    
        #scale surface
        x_new, y_new = scale_surface(x_imported, y_imported)
        #smooth surface
        window_size = 75
        y_smooth = smooth_surface(y_new, window_size) 
        
        #NOW, because U1 and U2 are very short, we need to 
        #shift them to start at x = 0. So:
        if (name == filenames[1]) or (name == filenames[2]): #if U1
            x_new -= min(x_new)
        
        #filter surface
        final_cond = filter_surface(x_new, y_smooth, n_nodes)
        array_of_surfaces[i] = final_cond
        i += 1
    return array_of_surfaces

all_surfaces = import_and_transform_all_surfaces(filenames)

all_surfaces[:, 0] = 0
all_surfaces = np.clip(all_surfaces,-999999,0)

#now set the initial condition
init_cond = all_surfaces[0, :] / 3
final_cond = all_surfaces[0, :]

br_elev_array[0, :] = init_cond[:]
dz = init_cond - final_cond

#now do a loop to populate every column with time series of bedrock elevation

exp_factor = 5 #this sets the shape of the exponential decay...
k = exp_factor * np.log(final_cond / init_cond) / 130000000 #calculate k at every point
t = np.arange(0, 130000000, 1000)
for i in range(len(x)): #for each element of x...
    br_elev_array[:, i] = dz[i]*np.exp(-k[i] * t) + final_cond[i]

br_elev_array[:, 0] = 0. #keep left-most node always at z=0

#save out array as npy binary
np.save('/scratch/cs00048/marine/sections/orange_section_R5/prepro/bedrock_elev_array.npy', br_elev_array)

phi = 0.56
z_star = 2830.
dx = 10000.
dt = 1000.
time_to_run = 130000000
qs_array = np.zeros(int(time_to_run / dt))

#duration of each layer from bottom to top
layer_starts = [0, 17000000, 30000000, 36000000, 49000000, 64000000, 100000000, 119000000]
layer_durations = [17000000, 13000000, 6000000, 13000000, 15000000, 36000000, 19000000, 11000000]

for i in range(8): #for each layer
    layer_start = int(layer_starts[i] / dt)
    layer_duration = int(layer_durations[i] / dt)
    
    tot_depth_top = all_surfaces[8, :]-all_surfaces[i + 1, :]
    tot_depth_bottom = all_surfaces[8, :]-all_surfaces[i, :]
    
    integral_at_upper_depth = phi * z_star * np.exp(-tot_depth_top/z_star) + tot_depth_top
    integral_at_lower_depth = phi * z_star * np.exp(-tot_depth_bottom/z_star) + tot_depth_bottom

    def_int = - (integral_at_upper_depth - integral_at_lower_depth)

    total_sed = np.sum(def_int) * dx #now have m2
    qs_over_time_period = total_sed / layer_durations[i] #17 million years #now have m2/yr
    qs_array[layer_start:layer_start + layer_duration] = qs_over_time_period
    
#save out array as npy binary
np.save('/scratch/cs00048/marine/sections/orange_section_R5/prepro/qs_array.npy', qs_array)

#save data!
#this is nrows=9 (bedrock + 8 other strat units), ncols = n_nodes. 
#It holds elevation of each layer at each node. NOT sed thickness, but absolute altitude.
np.save('/scratch/cs00048/marine/sections/orange_section_R5/prepro/all_surfaces.npy', all_surfaces)
print("seccessfully finished prepro")
