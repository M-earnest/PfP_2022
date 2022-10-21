# Using Python for neuroimaging data - NiBabel

The primary goal of this section is to become familiar with loading, modifying, saving, and visualizing neuroimages in Python. A secondary goal is to develop a conceptual understanding of the data structures involved, to facilitate diagnosing problems in data or analysis pipelines.

To these ends, we'll be exploring two libraries: [nibabel](http://nipy.org/nibabel/) and [nilearn](https://nilearn.github.io/). Each of these projects has excellent documentation. While this should get you started, it is well worth your time to look through these sites.

This notebook only covers nibabel, see the notebook [`image_manipulation_nilearn.ipynb`](image_manipulation_nilearn.ipynb) for more information about nilearn.

# Nibabel

Nibabel is a low-level Python library that gives access to a variety of imaging formats, with a particular focus on providing a common interface to the various **volumetric** formats produced by scanners and used in common neuroimaging toolkits.

 - NIfTI-1
 - NIfTI-2
 - SPM Analyze
 - FreeSurfer .mgh/.mgz files
 - Philips PAR/REC
 - Siemens ECAT
 - DICOM (limited support)

It also supports **surface** file formats

 - GIFTI
 - FreeSurfer surfaces, labels and annotations

**Connectivity**

 - CIFTI-2

**Tractography**

 - TrackViz .trk files

And a number of related formats.

**Note:** Almost all of these can be loaded through the `nibabel.load` interface.

## Setup

# Image settings
from nilearn import plotting
import pylab as plt
%matplotlib inline

import numpy as np
import nibabel as nb

## Loading and inspecting images in `nibabel`

# Load a functional image of subject 01
img = nb.load('/data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz')

# Let's look at the header of this file
print(img)

This data-affine-header structure is common to volumetric formats in nibabel, though the details of the header will vary from format to format.

### Access specific parameters

If you're interested in specific parameters, you can access them very easily, as the following examples show.

data = img.get_data()
data.shape

affine = img.affine
affine

header = img.header['pixdim']
header

Note that in the `'pixdim'` above contains the voxel resolution  (`4., 4., 3.999`), as well as the TR (`2.5`).

#### Aside
Why not just `img.data`? Working with neuroimages can use a lot of memory, so nibabel works hard to be memory efficient. If it can read some data while leaving the rest on disk, it will. `img.get_data()` reflects that it's doing some work behind the scenes.

#### Quirk

 - `img.get_data_dtype()` shows the type of the data on disk
 - `img.get_data().dtype` shows the type of the data that you're working with

These are not always the same, and not being clear on this [has caused problems](https://github.com/nipy/nibabel/issues/490). Further, modifying one does not update the other. This is especially important to keep in mind later when saving files.

print((data.dtype, img.get_data_dtype()))

### Data

The data is a simple numpy array. It has a shape, it can be sliced and generally manipulated as you would any array.

plt.imshow(data[:, :, data.shape[2] // 2, 0].T, cmap='Greys_r')
print(data.shape)

## Exercise 1:

Load the T1 data from subject 1. Plot the image using the same volume indexing as before. Also, print the shape of the data.

t1 = nb.load('/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz')
data = t1.get_data()
plt.imshow(data[:, :, data.shape[2] // 2].T, cmap='Greys_r')
print(data.shape)

# Work on solution here

### `img.orthoview()`

Nibabel has its own viewer, which can be accessed through **`img.orthoview()`**. This viewer scales voxels to reflect their size, and labels orientations.

**Warning:** `img.orthoview()` may not work properly on OS X.

#### Sidenote to plotting with `orthoview()`
As with other figures, f you initiated `matplotlib` with `%matplotlib inline`, the output figure will be static. If you use `orthoview()` in a normal IPython console, it will create an interactive window, and you can click to select different slices, similar to `mricron`. To get a similar experience in a jupyter notebook, use `%matplotlib notebook`. But don't forget to close figures afterward again or use `%matplotlib inline` again, otherwise, you cannot plot any other figures.

%matplotlib notebook
img.orthoview()

### Affine

The affine is a 4 x 4 numpy array. This describes the transformation from the voxel space (indices [i, j, k]) to the reference space (distance in mm (x, y, z)).

It can be used, for instance, to discover the voxel that contains the origin of the image:

x, y, z, _ = np.linalg.pinv(affine).dot(np.array([0, 0, 0, 1])).astype(int)

print("Affine:")
print(affine)
print
print("Center: ({:d}, {:d}, {:d})".format(x, y, z))

The affine also encodes the axis orientation and voxel sizes:

nb.aff2axcodes(affine)

nb.affines.voxel_sizes(affine)

nb.aff2axcodes(affine)

nb.affines.voxel_sizes(affine)

t1.orthoview()

### Header

The header is a nibabel structure that stores all of the metadata of the image. You can query it directly, if necessary:

t1.header['descrip']

But it also provides interfaces for the more common information, such as `get_zooms`, `get_xyzt_units`, `get_qform`, `get_sform`).

t1.header.get_zooms()

t1.header.get_xyzt_units()

t1.header.get_qform()

t1.header.get_sform()

Normally, we're not particularly interested in the header or the affine. But it's important to know they're there. And especially, to remember to copy them when making new images, so that derivatives stay aligned with the original image.

## `nib-ls`

Nibabel comes packaged with a command-line tool to print common metadata about any (volumetric) neuroimaging format nibabel supports. By default, it shows (on-disk) data type, dimensions and voxel sizes. 

!nib-ls /data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz

We can also inspect header fields by name, for instance, `descrip`:

!nib-ls -H descrip /data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz

## Creating and saving images

Suppose we want to save space by rescaling our image to a smaller datatype, such as an unsigned byte. To do this, we first need to take the data, change its datatype and save this new data in a new NIfTI image with the same header and affine as the original image.

# First, we need to load the image and get the data
img = nb.load('/data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz')
data = img.get_data()

# Now we force the values to be between 0 and 255
# and change the datatype to unsigned 8-bit
rescaled = ((data - data.min()) * 255. / (data.max() - data.min())).astype(np.uint8)

# Now we can save the changed data into a new NIfTI file
new_img = nb.Nifti1Image(rescaled, affine=img.affine, header=img.header)
nb.save(new_img, '/tmp/rescaled_image.nii.gz')

Let's look at the datatypes of the data array, as well as of the nifti image:

print((new_img.get_data().dtype, new_img.get_data_dtype()))

That's not optimal. Our data array has the correct type, but the on-disk format is determined by the header, so saving it with `img.header` will not do what we want. Also, let's take a look at the size of the original and new file.

orig_filename = img.get_filename()
!du -hL /tmp/rescaled_image.nii.gz $orig_filename

So, let's correct the header issue with the `set_data_dtype()` function:

img.set_data_dtype(np.uint8)

# Save image again
new_img = nb.Nifti1Image(rescaled, affine=img.affine, header=img.header)
nb.save(new_img, '/tmp/rescaled_image.nii.gz')
print((new_img.get_data().dtype, new_img.get_data_dtype()))

Perfect! Now the data types are correct. And if we look at the size of the image we even see that it got a bit smaller.

!du -hL /tmp/rescaled_image.nii.gz

# Conclusions

In the two notebooks about `nibabel` and `nilearn`, we've explored loading, saving and visualizing neuroimages, as well as how both packages can make some more sophisticated manipulations easy. At this point, you should be able to inspect and plot most images you encounter, as well as make modifications while preserving the alignment.