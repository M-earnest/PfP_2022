# MVPA and Searchlight with `nilearn`

In this section we will show how you can use `nilearn` to perform multivariate pattern analysis (MVPA) and a Searchlight analysis.


## `nilearn`

Although nilearn's visualizations are quite nice, its primary purpose was to facilitate machine learning in neuroimaging. It's in some sense the bridge between [nibabel](http://nipy.org/nibabel/) and [scikit-learn](http://scikit-learn.org/stable/). On the one hand, it reformats images to be easily passed to scikit-learn, and on the other, it reformats the results to produce valid nibabel images.

So let's take a look at a short multi-variate pattern analysis (MVPA) example.

**Note 1**: This section is heavily based on the [nilearn decoding tutorial](https://nilearn.github.io/auto_examples/plot_decoding_tutorial.html).  
**Note 2**: This section is not intended to teach machine learning, but to demonstrate a simple nilearn pipeline.

## Setup

from nilearn import plotting
%matplotlib inline
import numpy as np
import nibabel as nb

## Load machine learning dataset

Let's load the dataset we prepared in the previous notebook:

func = '/home/neuro/workshop/notebooks/data/dataset_ML.nii.gz'
!nib-ls $func

## Create mask

As we only want to use voxels in a particular region of interest (ROI) for the classification, let's create a function that returns a mask that either contains the only the brain, only the eyes or both:

from nilearn.image import resample_to_img, math_img
from scipy.ndimage import binary_dilation

def get_mask(mask_type):
    
    # Specify location of the brain and eye image
    brain = '/home/neuro/workshop/notebooks/data/templates/MNI152_T1_1mm_brain.nii.gz'
    eyes = '/home/neuro/workshop/notebooks/data/templates/MNI152_T1_1mm_eye.nii.gz'

    # Load region of interest
    if mask_type == 'brain':
        img_resampled = resample_to_img(brain, func)
    elif mask_type == 'eyes':
        img_resampled = resample_to_img(eyes, func)
    elif mask_type == 'both':
        img_roi = math_img("img1 + img2", img1=brain, img2=eyes)
        img_resampled = resample_to_img(img_roi, func)

    # Binarize ROI template
    data_binary = np.array(img_resampled.get_fdata()>=10, dtype=np.int8)

    # Dilate binary mask once
    data_dilated = binary_dilation(data_binary, iterations=1).astype(np.int8)

    # Save binary mask in NIfTI image
    mask = nb.Nifti1Image(data_dilated, img_resampled.affine, img_resampled.header)
    mask.set_data_dtype('i1')
    
    return mask

## Masking and Un-masking data

For the classification with `nilearn`, we need our functional data in a 2D, sample-by-voxel matrix. To get that, we'll select all the voxels defined in our `mask`.

from nilearn.plotting import plot_roi

anat = '/home/neuro/workshop/notebooks/data/templates/MNI152_T1_1mm.nii.gz'
mask = get_mask('both')
plot_roi(mask, anat, cmap='Paired', dim=-.5, draw_cross=False, annotate=False)

`NiftiMasker` is an object that applies a mask to a dataset and returns the masked voxels as a vector at each time point.

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask, standardize=False, detrend=False,
                     memory="nilearn_cache", memory_level=2)
samples = masker.fit_transform(func)
print(samples)

Its shape corresponds to the number of time-points times the number of voxels in the mask.

print(samples.shape)

To recover the original data shape (giving us a masked and z-scored BOLD series), we simply use the masker's inverse transform:

masked_epi = masker.inverse_transform(samples)

Let's now visualize the masked epi.

from nilearn.image import math_img
from nilearn.plotting import plot_stat_map

max_zscores = math_img("np.abs(img).max(axis=3)", img=masked_epi)
plot_stat_map(max_zscores, bg_img=anat, dim=-.5, cut_coords=[33, -20, 20],
              draw_cross=False, annotate=False, colorbar=False,
              title='Maximum Amplitude per Voxel in Mask')

# Simple MVPA Example

Multi-voxel pattern analysis (MVPA) is a general term for techniques that contrast conditions over multiple voxels. It's very common to use machine learning models to generate statistics of interest.

In this case, we'll use the response patterns of voxels in the mask to predict if the eyes were **closed** or **open** during a resting-state fMRI recording. But before we can do MVPA, we still need to specify two important parameters:

***First***, we need to know the label for each volume. From the last section of the [Machine Learning Preparation](machine_learning_preparation.ipynb) notebook, we know that we have a total of 384 volumes in our `dataset_ML.nii.gz` file and that it's always 4 volumes of the condition `eyes closed`, followed by 4 volumes of the condition `eyes open`, etc. Therefore our labels should be as follows:

labels = np.ravel([[['closed'] * 4, ['open'] * 4] for i in range(48)])
labels[:20]

***Second***, we need the `chunks` parameter. This variable is important if we want to do for example cross-validation. In our case we would ideally create 48 chunks, one for each subject. But because a cross-validation of 48 chunks takes very long, let's just create 6 chunks, containing always 8 subjects, i.e. 64 volumes:

chunks = np.ravel([[i] * 64 for i in range(6)])
chunks[:100]

One way to do cross-validation is the so called **Leave-one-out cross-validation**. This approach trains on `(n - 1)` chunks, and classifies the remaining chunk, and repeats this for every chunk, also called **fold**. Therefore, a 6-fold cross-validation is one that divides the whole data into 6 different chunks.

Now that we have the labels and chunks ready, we're only missing the classifier. In `Scikit-Learn`, there are [many to choose from](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html), let's start with the most well known, a linear support vector classifier (SVC).

# Let's specify the classifier
from sklearn.svm import LinearSVC
clf = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=25)

**Note:** The number of maximum iterations should ideally be much much bigger (around 1000), but was kept low here to reduce computation time.

Now, we're ready to train the classifier and do the cross-validation.

# Performe the cross validation (takes time to compute)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
cv_scores = cross_val_score(estimator=clf,
                            X=samples,
                            y=labels,
                            groups=chunks,
                            cv=LeaveOneGroupOut(),
                            n_jobs=-1,
                            verbose=1)

After the cross validation was computed we can extract the overall accuracy, as well as the accuracy for each individual fold (i.e. leave-one-out prediction). Mean (across subject) cross-validation accuracy is a common statistic for classification-based MVPA.

print('Average accuracy = %.02f percent\n' % (cv_scores.mean() * 100))
print('Accuracy per fold:', cv_scores, sep='\n')

**Wow, an average accuracy above 80%!!!** What if we use another classifier? Let's say a Gaussian Naive Bayes classifier?

# Let's specify a Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

cv_scores = cross_val_score(estimator=clf,
                            X=samples,
                            y=labels,
                            groups=chunks,
                            cv=LeaveOneGroupOut(),
                            n_jobs=1,
                            verbose=1)

print('Average accuracy = %.02f percent\n' % (cv_scores.mean() * 100))
print('Accuracy per fold:', cv_scores, sep='\n')

That was much quicker but less accurate. As was expected. What about a Logistic Regression classifier?

# Let's specify a Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', max_iter=25)

cv_scores = cross_val_score(estimator=clf,
                            X=samples,
                            y=labels,
                            groups=chunks,
                            cv=LeaveOneGroupOut(),
                            n_jobs=-1,
                            verbose=1)

print('Average accuracy = %.02f percent\n' % (cv_scores.mean() * 100))
print('Accuracy per fold:', cv_scores, sep='\n')

The prediction accuracy is again above **80%**, much better! But anyhow, how do we know if an accuracy value is significant or not? Well, one way to find this out is to do some permutation testing.

## Permutation testing

One way to test the quality of the prediction accuracy is to run the cross-validation multiple times, but permutate the labels of the volumes randomly. Afterward we can compare the accuracy value of the correct labels to the ones with the random / false labels. Luckily `Scikit-learn` already has a function that does this for us. So let's do it.

**Note**: We chose again the `GaussianNB` classifier to reduce the computation time per cross-validation. Additionally, we chose the number of iterations under `n_permutations` for the permutation testing very low, to reduce computation time as well. This value should ideally be much higher, at least 100.

# Let's chose again the linear SVC
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# Import the permuation function
from sklearn.model_selection import permutation_test_score

# Run the permuation cross-validation
null_cv_scores = permutation_test_score(estimator=clf,
                                        X=samples,
                                        y=labels,
                                        groups=chunks,
                                        cv=LeaveOneGroupOut(),
                                        n_permutations=25,
                                        n_jobs=-1,
                                        verbose=1)

So, let's take a look at the results:

print('Prediction accuracy: %.02f' % (null_cv_scores[0] * 100),
      'p-value: %.04f' % (null_cv_scores[2]),
      sep='\n')

Great! This means... Using resting-state fMRI images, we can predict if a person had their eyes open or closed with an accuracy significantly above chance level!

# Which region is driving the classification?

With a simple MVPA approach, we unfortunately don't know which regions are driving the classification accuracy. We just know that all voxels in the mask allow the classification of the two classes, but why? We need a better technique that tells us where in the head we should look.

There are many different ways to figure out which region is important for classification, but let us introduce you two different approaches that you can use in `nilearn`: `SpaceNet` and  `Searchlight`

## SpaceNet: decoding with spatial structure for better maps

SpaceNet implements spatial penalties which improve brain decoding power as well as decoder maps. The results are brain maps which are both sparse (i.e regression coefficients are zero everywhere, except at predictive voxels) and structured (blobby). For more detail, check out `nilearn`'s section about [SpaceNet](http://nilearn.github.io/decoding/space_net.html).

To train a SpaceNet on our data, let's first split the data into a training set (chunk 0-4) and a test set (chunk 5). 

# Create two masks that specify the training and the test set 
mask_test = chunks == 5
mask_train = np.invert(mask_test)

# Apply this sample mask to X (fMRI data) and y (behavioral labels)
from nilearn.image import index_img
X_train = index_img(func, mask_train)
y_train = labels[mask_train]

X_test = index_img(func, mask_test)
y_test = labels[mask_test]

Now we can fit the SpaceNet to our data with a TV-l1 penalty. ***Note*** again, that we reduced the number of `max_iter` to have a quick computation. In a realistic case this value should be around 1000.

from nilearn.decoding import SpaceNetClassifier

# Fit model on train data and predict on test data
decoder = SpaceNetClassifier(penalty='tv-l1',
                             mask=get_mask('both'),
                             max_iter=10,
                             cv=5,
                             standardize=True,
                             memory="nilearn_cache",
                             memory_level=2,
                             verbose=1)

decoder.fit(X_train, y_train)

Now that the `SpaceNet` is fitted to the training data. Let's see how well it does in predicting the test data.

# Predict the labels of the test data
y_pred = decoder.predict(X_test)

# Retrun average accuracy
accuracy = (y_pred == y_test).mean() * 100.
print("\nTV-l1  classification accuracy : %g%%" % accuracy)

Again above 80% prediction accuracy? But we wanted to know what's driving this prediction. So let's take a look at the fitting coefficients.

from nilearn.plotting import plot_stat_map, show
coef_img = decoder.coef_img_

# Plotting the searchlight results on the glass brain
from nilearn.plotting import plot_glass_brain
plot_glass_brain(coef_img, black_bg=True, colorbar=True, display_mode='lyrz', symmetric_cbar=False,
                 cmap='magma', title='graph-net: accuracy %g%%' % accuracy)

Cool! As expected the visual cortex (in the back of the head) and the eyes are driving the classification!

## Searchlight

Now the next question is: How high would the prediction accuracy be if we only take one small region to do the classification?

To answer this question we can use something that is called a **Searchlight** approach. The searchlight approach was first proposed by [Kriegeskorte et al., 2006](https://pdfs.semanticscholar.org/985c/ceaca8606443f9129616a26bbbbf952f2d7f.pdf). It is a widely used approach for the study of the fine-grained patterns of information in fMRI analysis. Its principle is relatively simple: a small group of neighboring features is extracted from the data, and the prediction function is instantiated on these features only. The resulting prediction accuracy is thus associated with all the features within the group, or only with the feature on the center. This yields a map of local fine-grained information, that can be used for assessing hypothesis on the local spatial layout of the neural code under investigation.

You can do a searchlight analysis in `nilearn` as follows:

from nilearn.decoding import SearchLight

# Specify the mask in which the searchlight should be performed
mask = get_mask('both')

# Specify the classifier to use
# Let's use again a GaussainNB classifier to reduce computation time
clf = GaussianNB()

# Specify the radius of the searchlight sphere  that will scan the volume
# (the bigger the longer the computation)
sphere_radius = 8  # in mm

Now we're ready to create the searchlight object.

# Create searchlight object
sl = SearchLight(mask,
                 process_mask_img=mask,
                 radius=sphere_radius,
                 estimator=clf,
                 cv=LeaveOneGroupOut(),
                 n_jobs=-1,
                 verbose=1)

# Run the searchlight algorithm
sl.fit(nb.load(func), labels, groups=chunks)

That took a while. So let's take a look at the results.

# First we need to put the searchlight output back into an MRI image
from nilearn.image import new_img_like
searchlight_img = new_img_like(func, sl.scores_)

Now we can plot the results. Let's plot it once on the glass brain and once from the side. For better interpretation on where the peaks are, let's set a minimum accuracy threshold of 60%.

from nilearn.plotting import plot_glass_brain
plot_glass_brain(searchlight_img, black_bg=True, colorbar=True, display_mode='lyrz',
                 threshold=0.6, cmap='magma', title='Searchlight Prediction Accuracy')

from nilearn.plotting import plot_stat_map
plot_stat_map(searchlight_img, cmap='magma', bg_img=anat, colorbar=True,
              display_mode='x', threshold=0.6, cut_coords=[0, 6, 12, 18],
              title='Searchlight Prediction Accuracy');

As expected and already seen before, the hotspots with high prediction accuracy are around the primary visual cortex (in the back of the head) and around the eyes.