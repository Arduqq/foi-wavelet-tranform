import numpy as np # Image processing
import pywt # Wavelet transformation
from matplotlib import pyplot as plt # Plotting
from scipy.stats import norm, entropy # Curve fitting and distance measurement

def compute_coefficients(im, level): 
    # Wavelet transformation
    # Discrete wavelet is applied {level} amount of times on the approximation coefficients
    c = pywt.wavedec2(im, 'db5', mode='periodization', level=level)
    c[0] /= np.abs(c[0]).max()
    # Normalization of the coefficients for plotting purposes
    for detail_level in range(level):
        c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
    return pywt.coeffs_to_array(c)

# Reference image
image = pywt.data.camera().astype(np.float32)
shape = image.shape

# Introducing noise through a gaussian distribution
gaussian = np.random.normal(0, 1, (shape[0],shape[1])) 
distorted_image = np.zeros(shape, np.float32)
distorted_image = image + gaussian


# Defining the decomposition level
max_lev = 4
label_levels = 3

# Iterating the decomposition process whilst drawing the plots
fig, axes = plt.subplots(4, 5, figsize=[14, 8])
for level in range(0, max_lev + 1):
    if level == 0:
        # Drawing the reference image, its distortiion and their pixel valuehistograms
        axes[0, 0].imshow(image, cmap=plt.cm.gray)
        axes[0, 0].set_title('Image')
        axes[0, 0].set_axis_off()

        axes[1, 0].hist(image.ravel(),256,[0,256])
        axes[1, 0].set_xlim(0,255)

        axes[2, 0].imshow(distorted_image, cmap=plt.cm.gray)
        axes[2, 0].set_title('Distorted Image')
        axes[2, 0].set_axis_off()
        
        axes[3, 0].hist(distorted_image.ravel(),256,[0,256])
        axes[3, 0].set_xlim(0,255)
        continue

    # Converting the coefficients in a processable array
    reference_coefficient_arr, _ = compute_coefficients(image, level)
    axes[0, level].set_title('Coefficients\n({} level)'.format(level))
    axes[0, level].imshow(abs(reference_coefficient_arr), cmap=plt.cm.gray)
    
    # Computing the coefficient histograms
    # Every histogram is the result of an iteration of the transformation for a different wavelet scaling
    h, b = np.histogram(reference_coefficient_arr, 100)
    hist_reference = h/reference_coefficient_arr.shape[0]
    
    # GGD estimation process to aproximate the reference coefficient histogram !!!
    hist_estimation_arr = np.zeros((1, 101))
    for frequency_coefficents in reference_coefficient_arr:
        # Fitting every histogram with a Gaussian distribution
        # Finding the two values capturing the curve
        mu, sigma = norm.fit(frequency_coefficents)
        _, bins, _ = axes[1, level].hist(frequency_coefficents, bins=100, density=True, alpha=0.5, histtype='step')
        # Applying the curve on the bins of the histogram
        hist_estimation = norm.pdf(bins, mu, sigma)
        axes[1, level].plot(bins, hist_estimation, 'k', linewidth=1)
        # Saving the curve for later KLD calculation
        hist_estimation_arr = np.vstack([hist_estimation_arr, [hist_estimation]])
    #axes[1, level].imshow(hist_estimation_arr)
    h, b = np.histogram(hist_estimation_arr, 100)
    hist_transmitted = h/hist_estimation_arr.shape[0]
    axes[1, level].set_title(f'KLD_approx: {entropy(hist_transmitted, hist_reference)}', size="7")
        
    # Repeating the coefficient acquisition process while drawing the corresponding histograms
    distorted_coefficient_arr, _ = compute_coefficients(distorted_image, level)
    axes[2, level].imshow(abs(distorted_coefficient_arr), cmap=plt.cm.gray)
    axes[3, level].hist(distorted_coefficient_arr, bins=100, density=True, histtype='step')
    h, _ = np.histogram(distorted_coefficient_arr, 100)
    hist_distorted = h/distorted_coefficient_arr.shape[0]

    # Computing the Kullback-Leibler Divergence which is included in the entropy() function
    axes[3, level].set_title(f'KLD_transmitted: {entropy(hist_distorted, hist_transmitted)} \n KLD_reference: {entropy(hist_distorted, hist_reference)}', size="7")

plt.tight_layout()
plt.show()