
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure  # skikit-image
import math
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.misc import derivative


def f(x):
    return voasmooth2[int(x)]

## Taken from reference
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


# Load data file
engineFile = open('engine.raw', 'rb')
engine_array = np.fromfile(engineFile, dtype=np.uint8)

# Get data in original form
# Store dimensions after reading
xx = 256
yy = 256
zz = 110
minIso = engine_array.min()
maxIso = engine_array.max()

# print("min iso: ", minIso)
# print("max iso: ", maxIso)

# let width = 492
# let height = 492
# let depth = 442

# Convert to 3D array
arr_3d = engine_array.reshape(xx, yy, zz)

#  Calculate gradient
gx, gy, gz = np.gradient(arr_3d, 50, 50, 50)
# print("arr_3d type: ", type(arr_3d))
# print("gx shape: ", gx.shape)

# G has gradient magnitude
g = np.array([[[0 for k in range(xx)] for j in range(yy)] for i in range(zz)]).transpose()

# Inverse of gradient magnitude
inv = np.array([[[0 for k in range(xx)] for j in range(yy)] for i in range(zz)]).transpose()

# For each voxel
for i in range(0, 255):
    for j in range(0, 255):
        for k in range(0, 110):
            ## Calculating the gradient magnitude
            g[i][j][k] = math.sqrt(gx[i][j][k] ** 2 + gy[i][j][k] ** 2 + gz[i][j][k] ** 2)
            if g[i][j][k] != 0:
                ## Calculating the inverse
                inv[i][j][k] = 1 / g[i][j][k]


## Calculating the Surface area
sSigma = np.array([0 for k in range(maxIso+1)])

 Surface area for isosurfaces
 for i in range(0, 256):
 #   Mesh for particular isovalue
     verts, faces, normals, values = measure.marching_cubes(arr_3d, i)
     sSigma[i] = measure.mesh_surface_area(verts, faces)

xaxis = np.array([k for k in range(256)])

# Gradient summation over an iso value
cg = np.array([0 for k in range(maxIso+1)])

#  For each iso value calculate the C(sigma)
for i in range(0, 255):
    for j in range(0, 255):
        for k in range(0, 110):
            iso = arr_3d[i][j][k]
            cg[iso] += inv[i][j][k]

cgsmooth = smooth(cg)
# plt.plot(cgsmooth[0:256], label="C(sigma)")  ## Alternative to smooth the curve https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

voa = cgsmooth[0:256] / sSigma

voasmooth = smooth(voa)
voasmooth1 = smooth(voasmooth)
voasmooth1 = smooth(voasmooth1)
voasmooth1 = smooth(voasmooth1)
voasmooth2 = smooth(voasmooth1)

differential = np.array([999.999 for k in range(voasmooth2.size)])
for i in range(voasmooth2.size):
     differential[i] = derivative(f, i, dx=1e-1)

minimaValues = np.array([0 for k in range(8)])
cnt = 0
minval =1e-4    ## It is present to remove the fluctuations in the curve and get more precise results. Value can be adjusted.
for i in range(1, voasmooth2.size-1):
    if differential[i] < 0 and differential[i + 1] > 0:     ## Taking only the minimas
        if abs(differential[i]-differential[i + 1]) < minval:
            minimaValues[cnt] = abs((i / voasmooth2.size) * 255)    ## Scaling to our data range of 255
            cnt += 1

newMin = np.array([0 for k in range(cnt)])

for i in range(cnt):
    newMin[i] = minimaValues[i]
print("final values: ", newMin)

## All the different plots
# plt.plot(voasmooth2[0:256], label="VOA" )
# plt.plot(voasmooth2, label="VOA" )
# plt.yscale("log")
# plt.title("VOA")
# plt.grid()
# plt.xticks(np.arange(0, voasmooth2.size, 10.0))
# plt.legend()
# plt.show()

