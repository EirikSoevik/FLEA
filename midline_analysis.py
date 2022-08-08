import numpy as np
from analysis import utils as autil
#import matplotlib
#matplotlib.use('Qt5Agg')
# matplotlib.use('QtAgg')
#import matplotlib.pyplot as plt
# import util
import os
from analysis import plotting as aplt


def main():
    """TODO:
    - constant midline length constraint?
    - improve midline extreme points
    - make sure midline follows fish and doesn't always end at 1
    - smooth spline
    - read all of "Optimal undulatory swimming..." Maertens et. al
    - how to implement the factorial C


    - find period from simple point, or by plotting
    - find multiple periods (2 or 3) -> find average
    - find wave freq from least square method
    - fft for removing noise (can do in time domain as well)
        mirroring
        fft
        cut out higher freq.
        inverse fft to reconstruct original signal
        remove mirroring
    """

    my_dir = "my_data/mask_output_Feb-17-2022_1216/masks/"
    dir_files = os.listdir(my_dir)

    save = True
    plotting = True
    spline_length = 15

    dict_list = autil.get_attributes(my_dir)

    coords_x, coords_y = autil.get_coords(my_dir=my_dir, dict_list=dict_list)

    centroid = autil.get_centroid(dict_list=dict_list)
    midlines_x, midlines_y = autil.get_midlines(my_dir=my_dir, dict_list=dict_list)
    T, N = midlines_x.shape
    mean_length, std_length = autil.get_average_midline_length(midlines_x, midlines_y,plotting) #plots

    new_midlines_x, new_midlines_y, new_centroid,\
    new_coords_x, new_coords_y = autil.axis_transformation(midlines_x, midlines_y, centroid, coords_x, coords_y)


    new_mean_length, new_std_length = autil.get_average_midline_length(new_midlines_x, new_midlines_y,plotting) # plots

    spline_x, my_splines = autil.midline_spline(new_midlines_x,new_midlines_y,spline_length,interp_kind='linear')

    f_x, f_y = autil.fourier_analysis(new_midlines_x,new_midlines_y, std_length)

    p_it, sample_match_x, sample_match_y = autil.find_position(midlines_x=midlines_x,midlines_y=midlines_y, sample_iteration=0,tol=0.1)

    if plotting == True:
        #aplt.midline_animation(x=new_midlines_x, y=new_midlines_y)
        #aplt.midline_animation_centroid(x=midlines_x,y=midlines_y,centroid=centroid)
        #aplt.body_midline_centroid(coords_x=coords_x, coords_y=coords_y, midlines_x=midlines_x, midlines_y=midlines_y,
        #                           centroid=centroid)
        #aplt.body_midline_centroid(coords_x=new_coords_x, coords_y=new_coords_y, midlines_x=new_midlines_x,
        #                           midlines_y=new_midlines_y, centroid=new_centroid)
        aplt.spline_plotting(my_splines,spline_x,new_coords_x,new_coords_y,new_midlines_x,new_midlines_y,new_centroid)

        aplt.fourier_plot(f_x, f_y, N)
        aplt.fourier_animation(f_x, f_y, N)
   # aplt.spline_plotting(my_splines, spline_x, new_coords_x, new_coords_y, new_midlines_x, new_midlines_y, new_centroid)
    aplt.spline_plot(spline_x,my_splines)

    print("Finished main")


if __name__ == "__main__":
    main()
