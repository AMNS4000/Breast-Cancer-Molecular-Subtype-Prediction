import numpy as np
import sys
import pydicom
import nibabel as nib
import os
import glob

def write_nifti(filename, vol, affine):
    """Write a nifti file with an affine matrix.
    """
    output = nib.Nifti1Image(vol.T, affine)
    nib.save(output, filename)


def create_affine(ipp, iop, ps):
    """Generate an affine matrix from DICOM IPP and IOP attributes.
    The ipp (ImagePositionPatient) parameter should an Nx3 array, and
    the iop (ImageOrientationPatient) parameter should be Nx6, where
    N is the number of DICOM slices in the series.
    The return values are the affine matrix and the pixdim.
    Note the the output will use DICOM anatomical coordinates:
    x increases towards the left, y increases towards the back.
    """
    # solve Ax = b where x is slope, intecept
    n = ipp.shape[0]
    A = np.column_stack([np.arange(n), np.ones(n)])
    x, r, rank, s = np.linalg.lstsq(A, ipp, rcond=None)
    # round small values to zero
    x[(np.abs(x) < 1e-6)] = 0.0
    vec = x[0,:] # slope
    pos = x[1,:] # intercept

    # pixel spacing should be the same for all image
    spacing = np.ones(3)
    spacing[1::-1] = ps[0,:]
    if np.sum(np.abs(ps - spacing[1::-1])) > spacing[0]*1e-6:
        sys.stderr.write("Pixel spacing is inconsistent!\n");

    # compute slice spacing, round to 7 decimal places
    spacing[2] = np.round(np.sqrt(np.sum(np.square(vec))), 7)

    # get the orientation
    iop_average = np.mean(iop, axis=0)
    u = iop_average[0:3]
    u /= np.sqrt(np.sum(np.square(u)))
    v = iop_average[3:6]
    v /= np.sqrt(np.sum(np.square(v)))

    # round small values to zero
    u[(np.abs(u) < 1e-6)] = 0.0
    v[(np.abs(v) < 1e-6)] = 0.0

    # create the matrix
    mat = np.eye(4)
    mat[0:3,0] = u*spacing[0]
    mat[0:3,1] = v*spacing[1]
    mat[0:3,2] = vec
    mat[0:3,3] = pos

    # check whether slice vec is orthogonal to iop vectors
    dv = np.dot(vec, np.cross(u, v))
    qfac = np.sign(dv)
    if np.abs(qfac*dv - spacing[2]) > 1e-6:
        sys.stderr.write("Non-orthogonal volume!\n");

    # compute the pixdim array, with qfac as the first element
    pixdim = np.hstack([np.array(qfac), spacing])

    return mat, pixdim


def convert_coords(vol, mat):
    """Convert a volume from DICOM coords to NIFTI coords or vice-versa.
    For DICOM, x increases to the left and y increases to the back.
    For NIFTI, x increases to the right and y increases to the front.
    The conversion is done in-place (volume and matrix are modified).
    """
    # the x direction and y direction are flipped
    convmat = np.eye(4)
    convmat[0,0] = -1.0
    convmat[1,1] = -1.0

    # apply the coordinate change to the matrix
    mat[:] = np.dot(convmat, mat)

    # look for x and y elements with greatest magnitude
    xabs = np.abs(mat[:,0])
    yabs = np.abs(mat[:,1])
    xmaxi = np.argmax(xabs)
    yabs[xmaxi] = 0.0
    ymaxi = np.argmax(yabs)

    # re-order the data to ensure these elements aren't negative
    # (this may impact the way that the image is displayed, if the
    # software that displays the image ignores the matrix).
    if mat[xmaxi,0] < 0.0:
        # flip x
        vol[:] = np.flip(vol, 2)
        mat[:,3] += mat[:,0]*(vol.shape[2] - 1)
        mat[:,0] = -mat[:,0]
    if mat[ymaxi,1] < 0.0:
        # flip y
        vol[:] = np.flip(vol, 1)
        mat[:,3] += mat[:,1]*(vol.shape[1] - 1)
        mat[:,1] = -mat[:,1]

    # eliminate "-0.0" (negative zero) in the matrix
    mat[mat == 0.0] = 0.0


def find_dicom_files(path):
    """Search for DICOM files at the provided location.
    """
    if os.path.isdir(path):
        # check for common DICOM suffixes
        for ext in ("*.dcm", "*.DCM", "*.dc", "*.DC", "*.IMG"):
            pattern = os.path.join(path, ext)
            files = glob.glob(pattern)
            if files:
                break
        # if no files with DICOM suffix are found, get all files
        if not files:
            pattern = os.path.join(path, "*")
            contents = glob.glob(pattern)
            files = [f for f in contents if os.path.isfile(f)]
    elif os.path.isfile(path):
        # if 'path' is a file (not a folder), return the file
        files = [path]
    else:
        sys.stderr.write("Cannot open %s\n" % (path,))
        return []

    return files


def load_dicom_series(files):
    """Load a series of dicom files and return a list of datasets.
    The resulting list will be sorted by InstanceNumber.
    """
    # start by sorting filenames lexically
    sorted_files = sorted(files)

    # create list of tuples (InstanceNumber, DataSet)
    dataset_list = []
    for f in files:
        ds = pydicom.dcmread(f)
        try:
            i = int(ds.InstanceNumber)
        except (AttributeError, ValueError):
            i = -1
        dataset_list.append( (i, ds) )

    # sort by InstanceNumber (the first element of each tuple)
    dataset_list.sort(key=lambda t: t[0])

    # get the dataset from each tuple
    series = [t[1] for t in dataset_list]

    return series


def dicom_to_volume(dicom_series):
    """Convert a DICOM series into a float32 volume with orientation.
    The input should be a list of 'dataset' objects from pydicom.
    The output is a tuple (voxel_array, voxel_spacing, affine_matrix)
    """
    # Create numpy arrays for volume, pixel spacing (ps),
    # slice position (ipp or ImagePositionPatient), and
    # slice orientation (iop or ImageOrientationPatient)
    n = len(dicom_series)
    shape = (n,) + dicom_series[0].pixel_array.shape
    vol = np.empty(shape, dtype=np.float32)
    ps = np.empty((n,2), dtype=np.float64)
    ipp = np.empty((n,3), dtype=np.float64)
    iop = np.empty((n,6), dtype=np.float64)

    for i, ds in enumerate(dicom_series):
        # create a single complex-valued image from real,imag
        image = ds.pixel_array
        try:
            slope = float(ds.RescaleSlope)
        except (AttributeError, ValueError):
            slope = 1.0
        try:
            intercept = float(ds.RescaleIntercept)
        except (AttributeError, ValueError):
            intercept = 0.0
        vol[i,:,:] = image*slope + intercept
        ps[i,:] = dicom_series[i].PixelSpacing
        ipp[i,:] = dicom_series[i].ImagePositionPatient
        iop[i,:] = dicom_series[i].ImageOrientationPatient

    # create nibabel-style affine matrix and pixdim
    # (these give DICOM LPS coords, not NIFTI RAS coords)
    affine, pixdim = create_affine(ipp, iop, ps)
    return vol, pixdim, affine
