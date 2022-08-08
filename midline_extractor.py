import os

import numpy as np

from util import utils
import time
import util.plotting as uplt
import math
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def rib_approximator_setup(my_file, outdir, save, plotting, N_midline):

    start_time = time.time()
    coords = np.load(outdir + my_file + "/" + my_file + ".npy")
    centroid = utils.find_centroid(coords=coords,outdir=outdir,save=False)
    midline_rib_approx = utils.midline_rib_approximation(coords, N_midline)
    midline_angles, midline_angles_change = utils.midline_angles(midline_rib_approx)
    #spline_x, my_spline = utils.midline_spline(midline=midline_rib_approx,spline_length=N_midline,interp_kind='quadratic')
    if save:
        np.save(outdir+my_file+"/coords.npy",coords)
        np.save(outdir+my_file+"/midline.npy", midline_rib_approx)
        np.save(outdir+my_file+"/midline_angles.npy", midline_angles)
        np.save(outdir+my_file+"/midline_angles_change.npy", midline_angles_change)
        np.save(outdir+my_file+"/centroid.npy",centroid)
        print("Saved " + outdir+my_file)
    end_time = time.time()


    if plotting:
        uplt.rib_approx(coords=coords, rib_midline=midline_rib_approx, midline_angles=midline_angles,midline_angles_change=midline_angles_change)
        #uplt.spline_plot(my_spline,coords,spline_x)
    print("Finished processing " + my_file + " in {sec:2.4f} seconds".format(sec=end_time - start_time))


if __name__ == "__main__":

    my_dir = "my_data/mask_output_Feb-17-2022_1216/masks/"
    dir_files = os.listdir(my_dir)
    save = False  # set to True to save all calculations, set to False to not save anything
    N_midline = 20
    plotting = True
    start_time = time.time()
    for my_file in dir_files:
        rib_approximator_setup(my_file=my_file, outdir=my_dir, save=save, plotting=plotting, N_midline=N_midline)
    end_time = time.time()
    print("Finished midline_extractor.py in {sec:2.3f} seconds".format(sec=end_time-start_time))
#     midline_rib_approx = utils.midline_rib_approximation(coords, N_midline, save=save)
#     midline_angles, midline_angles_change = utils.midline_angles(midline_rib_approx, save=save)
#
#     end_time = time.time()
#     print("Calculations took : " + str(end_time - start_time) + " seconds")
#
#     if plotting:
#         plt.plot(midline_angles * (180 / np.pi))
#         plt.show()
#         uplt.rib_approx(coords=coords, rib_midline=midline_rib_approx,midline_angles=midline_angles,midline_angles_change=midline_angles_change)
#         # uplt.midlines(midlines_unfiltered,midlines_filtered,midlines_interpolated,coords)
#
#     print("Finished midline_extractor.py")
