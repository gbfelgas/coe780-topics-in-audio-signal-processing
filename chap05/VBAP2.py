import numpy as np

def VBAP2(pan_dir):
    # function [gains] = VBAP3(pan_dir)
    # Author: V. Pulkki
    # Computes 3D VBAP gains for loudspeaker setup shown in Fig.6.4 
    # Change the lousdpeaker directions to match with your system, 
    # the directions are defined as azimuth elevation; pairs
    #
    #--------------------------------------------------------------------------
    # This source code is provided without any warranties as published in 
    # DAFX book 2nd edition, copyright Wiley & Sons 2011, available at 
    # http://www.dafx.de. It may be used for educational purposes and not 
    # for commercial applications without further permission.
    #--------------------------------------------------------------------------

    pi = np.pi

    ls_dirs = [30, -30, -90, -150, 150, 90]
    ls_dirs = np.array(ls_dirs)
    ls_num = ls_dirs.shape[0]
    ls_dirs = ls_dirs/180*pi
    
    # Panning direction in cartesian coordinates.
    panvec = np.array([np.cos(pan_dir/180*pi), np.sin(pan_dir/180*pi)])

    for i in range(ls_num):
        # Compute inverse of loudspeaker base matrix.
        if i == 5:
            lsmat = [[np.cos(ls_dirs[i]), np.sin(ls_dirs[i])], 
                     [np.cos(ls_dirs[0]), np.sin(ls_dirs[0])]]
        else:
            lsmat = [[np.cos(ls_dirs[i]), np.sin(ls_dirs[i])], 
                    [np.cos(ls_dirs[i+1]), np.sin(ls_dirs[i+1])]]
        # Compute unnormalized gains
        tempg = panvec.T @ np.linalg.inv(lsmat)
        # If gains nonnegative, normalize the gains and stop
        if min(tempg) > -0.001:
            g = np.zeros((ls_num,))
            g[i] = tempg[0]
            if i == 5:
                g[0] = tempg[1]
            else:
                g[(i%ls_num) + 1] = tempg[1]
            gains=g/(sum(g**2))**(1/2)
            return gains

if __name__ == "__main__":
	pandir = 45
	print(VBAP2(pandir))