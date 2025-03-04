from numpy import arange, sum, ix_
from numpy.fft import fft2, fftshift, ifftshift
from numpy.linalg import eps
from scipy.stats import dirichlet, poisson, norm
from image_formation import get_mean_image

def get_log_prior(inference, object, modulation_transfer_function_vectorized, raw_image_size_x, raw_image_size_y):

    padding_size = inference.padding_size

    val_range_x = arange(padding_size, padding_size+raw_image_size_x)
    val_range_y = arange(padding_size, padding_size+raw_image_size_y)
    mod_fft_image = abs(fftshift(fft2(ifftshift(object))))[ix_(val_range_x, val_range_y)].flatten + eps
    log_prior = dirichlet.logpdf(mod_fft_image / sum(mod_fft_image), modulation_transfer_function_vectorized + eps)
    return log_prior

def get_log_likelihood(object, shot_noise_image, inference, raw_image_with_padding, gain_map_with_padding, offset_map_with_padding,
                       error_map_with_padding, raw_img_size_x,raw_img_size_y):
    
    padding_size = inference.padding_size

    log_likelihood = 0.0
    mean_image = get_mean_image(object)
    val_range_x = arange(padding_size, padding_size+raw_img_size_x)
    val_range_y = arange(padding_size, padding_size+raw_img_size_y)

    log_likelihood += sum(poisson.logpmf(k=shot_noise_image[ix_(val_range_x, val_range_y)], 
                                         mu=mean_image[ix_(val_range_x, val_range_y)] + eps))
    
    log_likelihood += sum(norm.logpdf(raw_image_with_padding[ix_(val_range_x, val_range_y)], 
                                      loc=gain_map_with_padding[ix_(val_range_x, val_range_y)] * shot_noise_image[val_range_x, val_range_y] + 
                                          offset_map_with_padding[val_range_x, val_range_y], 
                                      scale=error_map_with_padding[val_range_x, val_range_y] + eps))
    return log_likelihood

def compute_full_log_posterior(object, shot_noise_image, inference, raw_image_with_padding, gain_map_with_padding, offset_map_with_padding,
                       error_map_with_padding, raw_img_size_x,raw_img_size_y, modulation_transfer_function_vectorized, raw_image_size_x, 
                       raw_image_size_y):
    
    log_likelihood = get_log_likelihood(object, shot_noise_image, inference, raw_image_with_padding, gain_map_with_padding, offset_map_with_padding,
                       error_map_with_padding, raw_img_size_x,raw_img_size_y)
    log_prior = get_log_prior(object, inference, object, modulation_transfer_function_vectorized, raw_image_size_x, raw_image_size_y)
    log_posterior = log_likelihood + log_prior
    
    return log_posterior