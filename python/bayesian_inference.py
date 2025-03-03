from scipy.special import gammaln
from math import log
from numpy import sum, abs, arange, dot, log, sqrt, exp, copyto
from numpy.linalg import eps
from scipy.stats import dirichlet, poisson, norm
from numpy.fft import fftshift, ifftshift
import random
from image_formation import get_widefield_image_ij

def log_Beta(a):
	z = sum(gammaln(a)) - gammaln(sum(a))
	return z

def log_Dirichlet(x, a):
	z = sum((a - 1.0) * log(x)) - log_Beta(a)
	return z

def log_Dirichlet_MTF(x, MTF_minus_one, log_Beta_MTF):
	x = MTF_minus_one * log(x)
	z = sum(x) - log_Beta_MTF
	return z

def get_log_likelihood_ij(i, j, mean_img_ij, shot_noise_img_ij, padding_size):
    
    log_likelihood = 0.0 
    for i in range(padding_size):
        for j in range(padding_size):
            log_likelihood += poisson.logpmf(mu=mean_img_ij[i, j] + eps, k=shot_noise_img_ij[i, j])
    
    return log_likelihood 

def get_log_prior_ij(FFT_var, img_ij, img_ij_abs, mod_fft_img_ij, i, j):
    
    fftshift(img_ij, FFT_var)
    img_ij_abs = abs(img_ij)
    mod_fft_img_ij = (img_ij_abs.flatten) / (sum(img_ij_abs)) + eps
    
# The following Gibbs sampler computes likelihood for the neighborhood only.

def sample_object_neighborhood(temperature, object, shot_noise_image, mean_img_ij, proposed_mean_img_ij, FFT_var, 
                               iFFT_var, img_ij_abs, mod_fft_img_ij, n_accepted, inference, sub_img_size_x, fft_plan,
                               sub_img_size_y, padding_size, half_padding_size, covariance_object, raw_image_size_x, raw_image_size_y,
                               sub_raw_image, sub_gain_map, sub_offset_map, sub_error_map):
	
    for j in arange(padding_size, padding_size+sub_img_size_y):
        for i in arange(padding_size, padding_size+sub_img_size_x):

            old_value = object[i, j]
            shot_noise_img_ij = shot_noise_image[i - half_padding_size: i + half_padding_size+1, j - half_padding_size: j + half_padding_size+1]
            obj_ij = object[i - half_padding_size: i + half_padding_size+1, j - half_padding_size: j + half_padding_size+1] 
            
            img_ij = ifftshift(obj_ij)
            FFT_var = dot(fft_plan, img_ij)
            
            old_log_prior = get_log_prior_ij(FFT_var, img_ij, img_ij_abs, mod_fft_img_ij, i, j)  

 			# FFT_var gets modified in the following function
            
            mean_img_ij = get_widefield_image_ij(mean_img_ij, FFT_var, iFFT_var, img_ij, i, j)
            
            old_log_likelihood = get_log_likelihood_ij(i, j, mean_img_ij, shot_noise_img_ij)
            
            old_log_posterior = old_log_likelihood + old_log_prior
            
            old_jac = log(old_value)

            proposed_value = norm.rvs(loc=log(old_value), scale=sqrt(covariance_object), size=1)[0]

            print(proposed_value)
            
            proposed_value = exp(proposed_value)
            
            object[i, j] = proposed_value

            proposed_obj_ij = object[i - padding_size:i + padding_size+1, j - padding_size:j + padding_size+1] 
            img_ij = ifftshift(proposed_obj_ij)
            FFT_var = dot(fft_plan, img_ij)
            
            proposed_log_prior = get_log_prior_ij(FFT_var, img_ij, img_ij_abs, mod_fft_img_ij, i, j)
            
            proposed_mean_img_ij = (proposed_mean_img_ij, FFT_var, iFFT_var, img_ij, i, j)
            
            proposed_log_likelihood = get_log_likelihood_ij(i, j, proposed_mean_img_ij, shot_noise_img_ij)
            
            proposed_jac = log(proposed_value)
            proposed_log_posterior = proposed_log_likelihood + proposed_log_prior
            
            log_hastings = (1.0/temperature) * (proposed_log_posterior - old_log_posterior) + proposed_jac - old_jac
            log_rand = log(random.random())
            
            if log_hastings > log_rand:
                copyto(mean_img_ij, proposed_mean_img_ij)
                n_accepted += 1
            else:
                object[i, j] = old_value

 
			# Sample Intermediate Expected Photon Counts on each Pixel
			# Choose the central pixel in the mean image
			# for expected photon count
            
            expected_photon_count = mean_img_ij[half_padding_size, half_padding_size] 
            old_log_likelihood = norm.logpdf(sub_raw_image[i, j], loc=sub_gain_map[i, j]*expected_photon_count + sub_offset_map[i, j], scale=sub_error_map[i, j] + eps)
            old_log_prior = poisson.logpmf(k = shot_noise_image[i, j], mu = expected_photon_count + eps)
            
            old_log_posterior = old_log_likelihood  + old_log_prior
            
            proposed_shot_noise_pixel = poisson.rvs(mu = shot_noise_image[i, j], size=1)[0]
            
            new_log_likelihood = norm.logpdf(sub_raw_image[i, j], loc=sub_gain_map[i, j]*proposed_shot_noise_pixel + sub_offset_map[i, j], scale=sub_error_map[i, j] + eps)
            
            new_log_prior = poisson.logpmf(k = proposed_shot_noise_pixel, mu = expected_photon_count + eps)
 			
            new_log_posterior = new_log_likelihood  + new_log_prior

            log_forward_proposal_probability = poisson.logpmf(k = proposed_shot_noise_pixel, mu = shot_noise_image[i, j] + eps)
            
            log_backward_proposal_probability = poisson.logpmf(k = shot_noise_image[i, j], mu = proposed_shot_noise_pixel + eps)
            
            log_hastings = (1.0/temperature)* (new_log_posterior - old_log_posterior) + log_backward_proposal_probability - log_forward_proposal_probability
   
            log_rand = log(random.random())
            if log_hastings > log_rand: 
                shot_noise_image[i, j] = proposed_shot_noise_pixel
                
    return object, shot_noise_image, n_accepted

def sample_object_neighborhood_MLE(temperature, object, shot_noise_image, mean_img_ij, proposed_mean_img_ij, FFT_var,
                                   iFFT_var, img_ij, img_ij_abs, mod_fft_img_ij, n_accepted, inference, sub_img_size_x, fft_plan,
                                   sub_img_size_y, padding_size, half_padding_size, covariance_object, raw_image_size_x, raw_image_size_y,
                                   sub_raw_image, sub_gain_map, sub_offset_map, sub_error_map):
       
    for j in arange(padding_size, padding_size+sub_img_size_y):
        for i in arange(padding_size, padding_size+sub_img_size_x):  
                
            old_value = object[i, j]
            shot_noise_img_ij = shot_noise_image[i - half_padding_size: i + half_padding_size+1, j - half_padding_size: j + half_padding_size+1]
            obj_ij = object[i - half_padding_size: i + half_padding_size+1, j - half_padding_size: j + half_padding_size+1]
            img_ij = ifftshift(obj_ij)
            FFT_var = dot(fft_plan, img_ij)

 			# FFT_var gets modified in the following function
            mean_img_ij = get_widefield_image_ij(mean_img_ij, FFT_var, iFFT_var, img_ij, i, j)
            
            old_log_likelihood = get_log_likelihood_ij(i, j, mean_img_ij, shot_noise_img_ij)
            
            old_log_posterior = old_log_likelihood 
            old_jac = log(old_value)

            proposed_value = norm.rvs(loc=log(old_value), scale=sqrt(covariance_object), size=1)[0]
            proposed_value = exp(proposed_value)
            object[i, j] = proposed_value

            proposed_obj_ij = object[i - padding_size:i + padding_size+1, j - padding_size:j + padding_size+1]
            img_ij = ifftshift(proposed_obj_ij)
            FFT_var = dot(fft_plan, img_ij)

            proposed_mean_img_ij = get_widefield_image_ij(proposed_mean_img_ij, FFT_var, iFFT_var, img_ij, i, j)
    		
            proposed_log_likelihood = get_log_likelihood_ij(i, j, proposed_mean_img_ij, shot_noise_img_ij)
            
            proposed_jac = log(proposed_value)
            proposed_log_posterior = proposed_log_likelihood 
            
            log_hastings = (1.0/temperature) * (proposed_log_posterior - old_log_posterior) + proposed_jac - old_jac
            
            if log_hastings > 0.0:
                mean_img_ij = proposed_mean_img_ij
                n_accepted += 1
            else:
                object[i, j] = old_value

 
			# Sample Intermediate Expected Photon Counts on each Pixel
			# Choose the central pixel in the mean image
			# for expected photon count
            expected_photon_count = mean_img_ij[half_padding_size, half_padding_size]
            old_log_likelihood = norm.logpdf(sub_raw_image[i, j], loc=sub_gain_map[i, j]*expected_photon_count + sub_offset_map[i, j], scale=sub_error_map[i, j] + eps)
            
            old_log_prior = poisson.logpmf(k = shot_noise_image[i, j], mu = expected_photon_count + eps)
            
            old_log_posterior = old_log_likelihood  + old_log_prior
            proposed_shot_noise_pixel = poisson.rvs(mu = shot_noise_image[i, j], size=1)[0]
            
            new_log_likelihood = norm.logpdf(sub_raw_image[i, j], loc=sub_gain_map[i, j]*proposed_shot_noise_pixel + sub_offset_map[i, j], scale=sub_error_map[i, j] + eps)
            
            new_log_prior = poisson.logpmf(k = proposed_shot_noise_pixel, mu = expected_photon_count + eps)

            new_log_posterior = new_log_likelihood  + new_log_prior

            log_forward_proposal_probability = poisson.logpmf(k = proposed_shot_noise_pixel, mu = shot_noise_image[i, j] + eps)

            log_backward_proposal_probability = poisson.logpmf(k = shot_noise_image[i, j], mu = proposed_shot_noise_pixel + eps)

            log_hastings = (1.0/temperature) * (new_log_posterior - old_log_posterior) + log_backward_proposal_probability - log_forward_proposal_probability
            
            if log_hastings > 0.0:
                shot_noise_image[i, j] = proposed_shot_noise_pixel
    
    return object, shot_noise_image, n_accepted
