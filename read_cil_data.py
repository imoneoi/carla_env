import h5py
import sys, os
import datetime
from PIL import Image
import numpy as np     # numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

from carla.planner.map import CarlaMap

import pdb

filenames = sys.argv[1:]

steerings = []
throttles = []
command_steering_data = {2.0:[],3.0:[],4.0:[],5.0:[]}
control_map = {2.0:"FOLLOW",3.0:"LEFT",4.0:"RIGHT",5.0:"STRAIGHT"}

infraction_rates = []
speeds = []

file_ct = 0
max_infraction = 0
max_infraction_file = None

camera = []
commands = []

fig, ax = plt.subplots(1,1)

town_map = CarlaMap("Town01", 0.1653, 50.0)
# map_img = mpimg.imread('figures/Town01.png')

# ax.imshow(map_img)
map_img = mpimg.imread('figures/Town01.png')


map_shape = town_map.map_image.shape
dims = np.shape(map_img)


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

total_uncertain_points = {2.0:[],3.0:[],4.0:[],5.0:[]}

for filename in filenames:
    
    print(filename)
    f = h5py.File(filename, 'r')
    '''dir_name = 'logs/extracted/'+filename.split('/')[1].split('.')[0]
    if not os.path.exists(dir_name):
       os.mkdir(dir_name)'''
    #fh = open(dir_name+'/actions.txt','w')

    # List all groups
    #print(f.keys())
    a_group_key = list(f.keys())
    print(a_group_key)
    # Get the data
    measurement_data = list(f[u'targets']) #list(f[u'agent_data'])
    img_data = list(f[u'rgb'])
    '''
    Steer, float
    Gas, float
    Brake, float
    Hand Brake, boolean
    Reverse Gear, boolean
    Steer Noise, float
    Gas Noise, float
    Brake Noise, float
    8- Position X, float
    9- Position Y, float
    10- Speed, float
    Collision Other, float
    Collision Pedestrian, float
    Collision Car, float
    Opposite Lane Inter, float
    Sidewalk Intersect, float
    Acceleration X,float
    Acceleration Y, float
    Acceleration Z, float
    Platform time, float
    Game Time, float
    Orientation X, float
    Orientation Y, float
    Orientation Z, float
    High level command, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
    Noise, Boolean ( If the noise, perturbation, is activated, (Not Used) )
    Camera (Which camera was used)
    Angle (The yaw angle for this camera)
    '''
    
    pose_x = []
    pose_y = []

    h_x = {2.0:[],3.0:[],4.0:[],5.0:[]}
    h_y = {2.0:[],3.0:[],4.0:[],5.0:[]}
    c_x = []
    c_y = []
    sizes = {2.0:[],3.0:[],4.0:[],5.0:[]}
    print('length:', len(measurement_data))
    ct = 0
    throttles = []
    
    position = town_map.convert_to_pixel( [float(measurement_data[0][8]),float(measurement_data[0][9]),0.0])
    last_position = position
    pdb.set_trace()
    
    for data_point in measurement_data:        

        if ct >= 5 and ct < len(measurement_data) - 1:
            last_position = position
            # true_last_position = (data_point[8],data_point[9])
            position = town_map.convert_to_pixel( [data_point[8],data_point[9],0.0])
            dist = np.linalg.norm(np.array(position)-np.array(last_position))
            # true_dist = np.linalg.norm(np.array((data_point[8], data_point[9])))
            # pdb.set_trace()
            if dist > 890 and len(pose_x) > 1:
                print('respawn')
                ax.scatter(pose_x[0],pose_y[0],c='r',s=5)
                ax.plot(pose_x, pose_y)
                pose_x=[]
                pose_y=[]
            throttles.append(data_point[1])
            pose_x.append(position[0])
            pose_y.append(position[1])
                
            #img = Image.fromarray(np.asarray(Image.fromarray(np.asarray(img_data[ct], dtype=np.uint8)), dtype=np.uint8))
            #imscatter(0.0,0.0,img,ax=ax2,zoom=1.2)
            # print 'estimated speed: ',data_point[9]*5*3.6
            
        ct += 1
        
    last_position = town_map.convert_to_pixel( [measurement_data[-3][8],measurement_data[-3][9],0.0])
    position = town_map.convert_to_pixel( [measurement_data[-2][8],measurement_data[-2][9],0.0])
    dist = np.linalg.norm(np.array(position)-np.array(last_position))
    if dist<650 and len(pose_x) > 5:
        ax.scatter(pose_x[0],pose_y[0],c='r',s=5)
        ax.plot(pose_x, pose_y, picker=5)
        pose_x=[]
        pose_y=[]

# # Get the size of the map_img
height, width, _ = map_img.shape

# Get the x-axis and y-axis limits of the existing plot
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# Set the extent of the map to match the scatter points range
ax.imshow(map_img, extent=[x_min, x_max, y_min, y_max])

plt.axis('off')
plt.tight_layout()
nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
plt.savefig('figures/cil_vis_trajs_{}.png'.format(nowTime))
plt.show()