import numpy as np

def VBAP3(pan_dir):
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

    loudspeakers = [[0, 0], [50, 0], [130, 0], [-130, 0],
                    [-50, 0],  [40, 45], [180, 45], [-40, 45]]
    loudspeakers = np.array(loudspeakers)
    ls_num = loudspeakers.shape[0]
    # Select the triangles of from the loudspeakers here
    ls_triangles = [[0, 1, 5], [1, 2, 5], [2, 3, 6], [3, 4, 7], [4, 0, 7],
                    [0, 5, 7], [2, 5, 6], [3, 6, 7], [5, 6, 7]]
    ls_triangles = np.array(ls_triangles)

    pi = np.pi
    
    panvec = np.zeros((3,))
    lsmat = np.zeros((3,3))
    # Go through all triangles
    for trip1 in range(ls_triangles.shape[0]):
        ls_trip1 = loudspeakers[ls_triangles[trip1,:],:]
        # Panning direction in cartesian coordinates
        cosE = np.cos(pan_dir[1]/180*pi)
        panvec[0] = np.cos(pan_dir[0]/180*pi)*cosE
        panvec[1] = np.sin(pan_dir[0]/180*pi)*cosE
        panvec[2] = np.sin(pan_dir[1]/180*pi)
        # Loudspeaker base matrix for current triangle.
        for i in range(3):
            cosE = np.cos(ls_trip1[i, 1]/180*pi)
            lsmat[i,0] = np.cos(ls_trip1[i,0]/180*pi)*cosE
            lsmat[i,1] = np.sin(ls_trip1[i,0]/180*pi)*cosE
            lsmat[i,2] = np.sin(ls_trip1[i,1]/180*pi)

        tempg = panvec.T @ np.linalg.inv(lsmat) # Gain factors for current triangle.
        
        # If gains nonnegative, normalize g and stop computation
        if np.min(tempg) > -0.01:
            tempg = tempg / (np.sum(tempg**2))**(1/2)
            gains = np.zeros((1,ls_num))
            gains[0,ls_triangles[trip1,:]] = tempg
            return gains

if __name__ == "__main__":
	pandir = [0, 30]
	print(VBAP3(pandir))