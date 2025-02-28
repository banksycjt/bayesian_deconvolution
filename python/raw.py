import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
from math import pi, sqrt, ceil
import multiprocessing
from scipy.special import j1
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.stats import dirichlet, poisson, norm
from scipy.special import gammaln
import random

class OpticalSystem:
    def __init__(self, psf_type, numerical_aperture, magnification, light_wavelength, abbe_diffraction_limit, f_diffraction_limit, sigma):
        self.psf_type = psf_type
        self.numerical_aperture = numerical_aperture
        self.magnification = magnification
        self.light_wavelength = light_wavelength
        self.abbe_diffraction_limit = abbe_diffraction_limit
        self.f_diffraction_limit = f_diffraction_limit
        self.sigma = sigma

class Camera:
    def __init__(self, camera_pixel_size, physical_pixel_size, dx, gain, offset, noise, noise_maps_available):
        self.camera_pixel_size = camera_pixel_size
        self.physical_pixel_size = physical_pixel_size
        self.dx = dx
        self.gain = gain
        self.offset = offset
        self.noise = noise
        self.noise_maps_available = noise_maps_available

class Inference:
    def __init__(self, padding_size, half_padding_size, covariance_object, conc_parameter, n_procs_per_dim_x, n_procs_per_dim_y, total_draws,
                  chain_burn_in_period, chain_starting_temperature, chain_time_constant, annealing_starting_temperature, annealing_time_constant, 
                  annealing_burn_in_period, annealing_frequency, averaging_frequency, plotting_frequency):
        self.padding_size = padding_size
        self.half_padding_size = half_padding_size
        self.covariance_object = covariance_object
        self.conc_parameter = conc_parameter
        self.n_procs_per_dim_x = n_procs_per_dim_x
        self.n_procs_per_dim_y = n_procs_per_dim_y
        self.total_draws = total_draws
        self.chain_burn_in_period = chain_burn_in_period
        self.chain_starting_temperature = chain_starting_temperature
        self.chain_time_constant = chain_time_constant
        self.annealing_starting_temperature = annealing_starting_temperature
        self.annealing_time_constant = annealing_time_constant
        self.annealing_burn_in_period = annealing_burn_in_period
        self.annealing_frequency = annealing_frequency
        self.averaging_frequency = averaging_frequency
        self.plotting_frequency = plotting_frequency

def main():

    optical_system, camera, inference, raw_image_path = input_parameter()
    raw_image_size_x,raw_image_size_y = input_data(raw_image_path)
    print("Adding processors...")
    pool = multiprocessing.Pool(processes=inference.n_procs_per_dim_x*inference.n_procs_per_dim_y)
    print("Processors added.")
    sub_raw_img_size_x = raw_image_size_x/inference.n_procs_per_dim_x
    sub_raw_img_size_y = raw_image_size_y/inference.n_procs_per_dim_y

    # Generate PSF
    generate_psf(raw_image_size_x, raw_image_size_y, inference, optical_system, camera)


    return None    

def input_parameter():
    
    raw_image_path = ""+"raw_image.tif"

    # Optical Parameters
    psf_type = "airy_disk" 
    numerical_aperture = 1.3
    magnification = 100.0
    light_wavelength = 0.660# In micrometers
    abbe_diffraction_limit = light_wavelength/(2*numerical_aperture) # optical resolution in micrometers
    f_diffraction_limit = 1/abbe_diffraction_limit # Diffraction limit in k-space
    sigma = sqrt(2.0)/(2.0*pi)*light_wavelength/numerical_aperture # Standard Deviation in PSF

    # Camera Parameters
    camera_pixel_size = 6.5 # In micrometers
    physical_pixel_size = camera_pixel_size/magnification #in micrometers
    dx = physical_pixel_size #physical grid spacing
    gain = 1.957
    offset = 100.0
    noise = 2.3
    noise_maps_available = False

    # Inference Parameters
    padding_size = 4*ceil(abbe_diffraction_limit/physical_pixel_size) 
    half_padding_size = padding_size/2
    covariance_object  = 0.5
    conc_parameter = 1.0

    # Number of Processors Available to use 
    n_procs_per_dim_x = 2
    n_procs_per_dim_y = 2
    total_draws = 50000
    chain_burn_in_period = 1000
    chain_starting_temperature = 1000.0
    chain_time_constant = 50.0

    annealing_starting_temperature = 100.0
    annealing_time_constant = 30.0
    annealing_burn_in_period = 300
    annealing_frequency = annealing_burn_in_period + 50
    averaging_frequency = 10
    plotting_frequency = 10

    optical_system = OpticalSystem(psf_type, numerical_aperture, magnification, light_wavelength, abbe_diffraction_limit, f_diffraction_limit, sigma)
    camera = Camera(camera_pixel_size, physical_pixel_size, dx, gain, offset, noise, noise_maps_available)
    inference = Inference(padding_size, half_padding_size, covariance_object, conc_parameter, n_procs_per_dim_x, n_procs_per_dim_y, total_draws,
                           chain_burn_in_period, chain_starting_temperature, chain_time_constant, annealing_starting_temperature, 
                           annealing_time_constant, annealing_burn_in_period, annealing_frequency, averaging_frequency, plotting_frequency)

    return optical_system, camera, inference, raw_image_path

def input_data(raw_image_path):

    input_raw_image = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
    input_raw_image = input_raw_image.astype(np.float64)    

    raw_image_size_x = input_raw_image.shape[0]
    raw_image_size_y = input_raw_image.shape[1]
    gain = 1.957
    offset = 100.0
    noise = 2.3
    offset_map, error_map, gain_map = get_camera_calibration_data(gain, offset, noise, raw_image_size_x, raw_image_size_y)
    raw_image_with_padding = add_padding_reflective_BC(input_raw_image)
    gain_map_with_padding = add_padding_reflective_BC(gain_map)
    offset_map_with_padding = add_padding_reflective_BC(offset_map)
    error_map_with_padding = add_padding_reflective_BC(error_map)

    median_photon_count = np.ones((raw_image_size_x,raw_image_size_y))* np.median(abs((input_raw_image - np.ones((raw_image_size_x,raw_image_size_y))*np.median(offset_map_with_padding)) 
                                                                                       / (np.ones((raw_image_size_x,raw_image_size_y))*np.median(gain_map_with_padding))))
    return raw_image_size_x,raw_image_size_y

def get_camera_calibration_data(gain, offset, noise, raw_image_size_x, raw_image_size_y):

    offset_map = offset * np.ones(raw_image_size_x, raw_image_size_y)
    gain_map = gain * np.ones(raw_image_size_x, raw_image_size_y)
    error_map = noise * np.ones(raw_image_size_x, raw_image_size_y)

    return offset_map, error_map, gain_map

def add_padding_reflective_BC(input_img, inference):

    padding_size = inference.padding_size
    input_img = np.array(input_img)
    size_x = np.size(input_img)[0]
    size_y = np.size(input_img)[1]
    
    img = np.zeros(3*size_x, 3*size_y)
    img[size_x:-size_x, size_y:-size_y] = input_img 
    img[:size_x, size_y:-size_y]= np.flip(input_img,0)
    img[-size_x:, size_y:-size_y]= np.flip(input_img,0)
    img[size_x:-size_x, :size_y]= np.flip(input_img,1)
    img[size_x:-size_x, -size_y:]= np.flip(input_img,1)
    img[:size_x, :size_y]= np.flip(np.flip(input_img,0),1)
    img[:size_x, -size_y:]= np.flip(np.flip(input_img,0),1)
    img[-size_x:, :size_y]= np.flip(np.flip(input_img,0),1)
    img[-size_x:, -size_y:]= np.flip(np.flip(input_img,0),1)
    return img[size_x-padding_size:-size_x+padding_size, size_y-padding_size:-size_y+padding_size]

def apply_reflective_BC_object(object, intermediate_img, padding_size):
    size_x = np.size(object)[0] - 2*padding_size
    size_y = np.size(object)[1] - 2*padding_size

    object_padding = object[padding_size:-padding_size, padding_size:-padding_size]
    
    intermediate_img[size_x:-size_x, size_y:-size_y] = object_padding 
    intermediate_img[:size_x, size_y:-size_y] = object_padding[::-1, :] 
    intermediate_img[-size_x:, size_y:-size_y] = object_padding[::-1, :] 
    intermediate_img[size_x:-size_x, :size_y] = object_padding[:, ::-1] 
    intermediate_img[size_x:-size_x, -size_y:] = object_padding[:, ::-1] 
    intermediate_img[:size_x, :size_y] = object_padding[::-1, ::-1]
    intermediate_img[:size_x, -size_y:] = object_padding[::-1, ::-1]
    intermediate_img[-size_x:, :size_y] = object_padding[::-1, ::-1]
    intermediate_img[-size_x:, -size_y:] = object_padding[::-1, ::-1] 
    
    object = intermediate_img[size_x-padding_size:-size_x+padding_size, size_y-padding_size:-size_y+padding_size]
    
    return None

def apply_reflective_BC_shot(shot_noise_image, intermediate_img, padding_size):
    
    size_x = np.size(shot_noise_image)[1] - 2*padding_size
    size_y = np.size(shot_noise_image)[2] - 2*padding_size
    shot_noise_padding = shot_noise_image[padding_size:-padding_size, padding_size:-padding_size]
    
    intermediate_img[size_x:-size_x, size_y:-size_y] = shot_noise_padding 
    intermediate_img[:size_x, size_y:-size_y] = shot_noise_padding[::-1, :] 
    intermediate_img[-size_x:, size_y:-size_y] = shot_noise_padding[::-1, :] 
    intermediate_img[size_x:-size_x, :size_y] = shot_noise_padding[:, ::-1] 
    intermediate_img[size_x:-size_x, -size_y:] = shot_noise_padding[:, ::-1] 
    intermediate_img[:size_x, :size_y] = shot_noise_padding[::-1, ::-1]
    intermediate_img[:size_x, -size_y:] = shot_noise_padding[::-1, ::-1]
    intermediate_img[-size_x:, :size_y] = shot_noise_padding[::-1, ::-1]
    intermediate_img[-size_x:, -size_y:] = shot_noise_padding[::-1, ::-1] 
    
    shot_noise_image = intermediate_img[size_x-padding_size:-size_x+padding_size, size_y-padding_size:-size_y+padding_size]
    
    return None

def incoherent_PSF_airy_disk(x_c, x_e, light_wavelength, numerical_aperture):
    x_c = np.array(x_c)
    x_e = np.array(x_e)
    f_number = 1/(2*numerical_aperture) ##approx
    return (jinc(np.linalg.norm(x_c - x_e)/(light_wavelength*f_number)))^2

def incoherent_PSF_gaussian(x_c, x_e, sigma):
    x_c = np.array(x_c)
    x_e = np.array(x_e)
    return math.exp(-np.linalg.norm(x_c-x_e)^2/(2.0*sigma^2)) /(math.sqrt(2.0*math.pi) * sigma)^(np.shape(x_e)[0])

def jinc(x):
    '''
    计算 Jinc 函数：jinc(x) = J₁(x) / x
    注意：当 x = 0 时，J₁(x) / x 的值为 1/2。
    '''
    if x == 0:
        return 0.5  # 数学上，jinc(0) = 1/2
    else:
        return j1(x) / x

def MTF_air_disk(f_vector, light_wavelength, numerical_aperture):
		f_number = 1/(2*numerical_aperture) ##approx
		highest_transmitted_frequency = 1.0 / (light_wavelength*f_number) 
		norm_f = np.linalg.norm(f_vector) / highest_transmitted_frequency
		if norm_f < 1.0:
			mtf = 2.0/np.pi * (math.acos(norm_f) - norm_f*math.sqrt(1 - norm_f^2))
		elif norm_f >= 1.0:
			mtf = 0.0
		return mtf

def generate_psf(raw_image_size_x, raw_image_size_y, inference, optical_system, camera):

    grid_physical_1D_x = camera.dx * np.arange(-(raw_image_size_x/2 + inference.padding_size),(raw_image_size_x/2 + inference.padding_size)) # in micrometers
    grid_physical_1D_y = camera.dx * np.arange(-(raw_image_size_y/2 + inference.padding_size),(raw_image_size_y/2 + inference.padding_size)) # in micrometers
    
    grid_physical_length_x = (raw_image_size_x + 2*inference.padding_size - 1)*camera.dx			
    grid_physical_length_y = (raw_image_size_y + 2*inference.padding_size - 1)*camera.dx

    df_x = 1/(grid_physical_length_x) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 
    df_y = 1/(grid_physical_length_y) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 

    f_corrected_grid_1D_x = df_x * np.arange(-(raw_image_size_x/2 + inference.padding_size),(raw_image_size_x/2 + inference.padding_size)) # in units of micrometer^-1
    f_corrected_grid_1D_y = df_y * np.arange(-(raw_image_size_y/2 + inference.padding_size),(raw_image_size_y/2 + inference.padding_size)) # in units of micrometer^-1

    mtf_on_grid = np.zeros(raw_image_size_x+2*inference.padding_size, raw_image_size_y+2*inference.padding_size)
    psf_on_grid = np.zeros(raw_image_size_x+2*inference.padding_size, raw_image_size_y+2*inference.padding_size)

    if OpticalSystem.psf_type == "airy_disk":
        
        for j in range(raw_image_size_y + 2*inference.padding_size):
            for i in range(raw_image_size_y + 2*inference.padding_size):
                x_e = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
                psf_on_grid[i, j] =  incoherent_PSF_airy_disk([0.0, 0.0], x_e, optical_system.light_wavelength, optical_system.numerical_aperture)
 		
        normalization = np.sum(psf_on_grid) * camera.dx^2
        psf_on_grid = psf_on_grid / normalization
        intermediate_img = fftshift(fft(ifftshift(psf_on_grid)))

        for j in range(raw_image_size_y + 2*inference.padding_size):
            for i in range(raw_image_size_x + 2*inference.padding_size):
                mtf_on_grid[i, j] = MTF_air_disk([f_corrected_grid_1D_x[i], f_corrected_grid_1D_y[j]], optical_system.light_wavelength, optical_system.numerical_aperture)
                if mtf_on_grid[i, j] == 0.0:
                    intermediate_img[i, j] = 0.0 * intermediate_img[i, j]

        FFT_point_spread_function = ifftshift(intermediate_img) 

    elif OpticalSystem.psf_type == "gaussian":

        for j in range(raw_image_size_y+ 2*inference.padding_size):
            for i in range(raw_image_size_x + 2*inference.padding_size):
                x_e = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
                psf_on_grid[i, j] =  incoherent_PSF_gaussian([0.0, 0.0], x_e)
        
        FFT_point_spread_function = fft(ifftshift(psf_on_grid))
        
        
    modulation_transfer_function = abs(fftshift(FFT_point_spread_function))[inference.padding_size:-inference.padding_size, inference.padding_size:-inference.padding_size] 
    modulation_transfer_function_vectorized = (modulation_transfer_function.flatten()) / sum(modulation_transfer_function)

    psf_on_grid = 0
    mtf_on_grid = 0
    grid_physical_1D_x = 0
    grid_physical_1D_y = 0
    f_corrected_grid_1D_x = 0
    f_corrected_grid_1D_y = 0
    intermediate_img = 0

def get_widefield_image(object, camera, FFT_point_spread_function):
    FFT_illuminated_object = fft(ifftshift(object)) * camera.dx^2
    FFT_final = FFT_point_spread_function * FFT_illuminated_object
    image = abs(fftshift(ifft(FFT_final)).real)

    return image

def get_mean_image(object):
    final_image = get_widefield_image(object)
    return final_image

def get_log_prior(inference, object, modulation_transfer_function_vectorized, raw_image_size_x, raw_image_size_y):
    val_range_x = np.arange(inference.padding_size,inference.padding_size+raw_image_size_x)
    val_range_y = np.arange(inference.padding_size,inference.padding_size+raw_image_size_y)
    mod_fft_image = abs(fftshift(fft(ifftshift(object))))[val_range_x, val_range_y].flatten + np.finfo(np.float64).eps
    log_prior = dirichlet.logpdf(mod_fft_image / sum(mod_fft_image), modulation_transfer_function_vectorized + np.finfo(np.float64).eps)
    return log_prior

def get_log_likelihood(object, shot_noise_image, inference, raw_image_with_padding, gain_map_with_padding, offset_map_with_padding,
                       error_map_with_padding, raw_img_size_x,raw_img_size_y):
    log_likelihood = 0.0
    mean_image = get_mean_image(object)
    val_range_x = np.arange(inference.padding_size,inference.padding_size+raw_img_size_x)
    val_range_y = np.arange(inference.padding_size,inference.padding_size+raw_img_size_y)

    log_likelihood += np.sum(poisson.logpmf(k=shot_noise_image[val_range_x, val_range_y], 
                                            mu=mean_image[val_range_x, val_range_y] + np.finfo(np.float64).eps))
    
    log_likelihood += np.sum(norm.logpdf(raw_image_with_padding[val_range_x, val_range_y], 
                                      loc=gain_map_with_padding[val_range_x, val_range_y] * shot_noise_image[val_range_x, val_range_y] + offset_map_with_padding[val_range_x, val_range_y], 
                                      scale=error_map_with_padding[val_range_x, val_range_y] + np.finfo(float).eps))
    return log_likelihood

def compute_full_log_posterior(object, shot_noise_image):
    
    log_likelihood = get_log_likelihood(object, shot_noise_image)
    log_prior = get_log_prior(object)
    log_posterior = log_likelihood + log_prior
    
    return log_posterior

def save_data(current_draw, mcmc_log_posterior, object, shot_noise_image, object_mean, averaging_counter, 
              inference, raw_image_size_x,raw_image_size_y, working_directory):
    
    gray_image = cv2.cvtColor(object[inference.padding_size:inference.padding_size + raw_image_size_x, 
                                     inference.padding_size:inference.padding_size + raw_image_size_y], 
                                     cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(working_directory + "inferred_object_" + str(averaging_counter) + ".tif", gray_image)
	
    gray_image = cv2.cvtColor(object_mean[inference.padding_size:inference.padding_size + raw_image_size_x, 
                                          inference.padding_size:inference.padding_size + raw_image_size_y], 
                                          cv2.COLOR_BGR2GRAY)

    cv2.imwrite(working_directory + "mean_inferred_object_" + str(averaging_counter) + ".tif", gray_image)
	
    return None

def plot_data(current_draw, object, mean_object, shot_noise_image, log_posterior, inference, raw_image_size_x, raw_image_size_y, working_directory, raw_image_with_padding):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # 绘制第一张图
    axs[0, 0].imshow(object[inference.padding_size:inference.padding_size + raw_image_size_x, inference.padding_size:inference.padding_size + raw_image_size_y],
                     cmap='gray', origin='lower')
    axs[0, 0].set_title("Current Sample", fontsize=12)
    axs[0, 0].axis('off')  # 关闭坐标轴
    
    # 绘制第二张图
    axs[0, 1].imshow(mean_object[inference.padding_size:inference.padding_size + raw_image_size_x, inference.padding_size:inference.padding_size + raw_image_size_y],
                     cmap='gray', origin='lower')
    axs[0, 1].set_title("Mean", fontsize=12)
    axs[0, 1].axis('off')

    # 绘制第三张图

    axs[1, 0].plot(log_posterior[:current_draw], color='gray')
    axs[1, 0].set_title("log(Posterior)", fontsize=12)
    axs[1, 0].grid(True)
    axs[1, 0].legend(False)

    # 绘制第四张图
    axs[1, 1].imshow(raw_image_with_padding[inference.padding_size:inference.padding_size + raw_image_size_x, inference.padding_size:inference.padding_size + raw_image_size_y],
                     cmap='gray', origin='lower')
    axs[1, 1].set_title("Raw Image", fontsize=12)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
	
    return None

def get_sub_image(img, im, ip, jm, jp):
	sub_img = img[im:ip, jm:jp]
	return sub_img

def chunk_images(raw_image_size_x, raw_image_size_y, sub_raw_image_size_x, sub_raw_image_size_y, n_procs_per_dim_x, n_procs_per_dim_y,
                 padding_size, raw_image_with_padding, gain_map_with_padding, offset_map_with_padding, error_map_with_padding, processes):
    

    # Cartesian coordinates for each processor
    i_procs = (processes-1) % n_procs_per_dim_x
    j_procs = (processes-2 - i_procs) / n_procs_per_dim_x

    # Boundary coordinates of chunks or sub raw images in x an y direction
    im_raw = i_procs*sub_raw_image_size_x + 1
    ip_raw = 2*padding_size + (i_procs+1) * sub_raw_image_size_x
    jm_raw = j_procs*sub_raw_image_size_y + 1
    jp_raw = 2*padding_size + (j_procs+1) * sub_raw_image_size_y

    sub_img_size_x = ip_raw - im_raw + 1 - 2*padding_size
    sub_img_size_y = jp_raw - jm_raw + 1 - 2*padding_size



    image = raw_image_with_padding
    sub_raw_image = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

    image = gain_map_with_padding
    sub_gain_map = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

    image = offset_map_with_padding
    sub_offset_map = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

    image = error_map_with_padding
    sub_error_map = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

    image = 0

    return sub_raw_image, sub_gain_map, sub_offset_map, sub_error_map, sub_img_size_x, sub_img_size_y

def psf_for_inference(camera, optical_system, inference, raw_image_size_x, raw_image_size_y, processes, psf_type):
    
    grid_physical_1D_ij = camera.dx * np.arange(-inference.padding_size, inference.padding_size) # in micrometers
    grid_physical_length_ij = 2.0*inference.padding_size*camera.dx

    df_ij = 1/(grid_physical_length_ij) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 
    f_corrected_grid_1D_ij = df_ij * np.arange(-inference.padding_size, inference.padding_size)

    mtf_on_grid_ij = np.zeros((2*inference.padding_size+1, 2*inference.padding_size+1))
    psf_on_grid_ij = np.zeros((2*inference.padding_size+1, 2*inference.padding_size+1))

    if psf_type == "airy_disk":
        for j in range(2*inference.padding_size+1):
               for i in range(2*inference.padding_size+1):
                     x_e = [grid_physical_1D_ij[i], grid_physical_1D_ij[j]]
                     psf_on_grid_ij[i, j] =  incoherent_PSF_airy_disk([0.0, 0.0], x_e, optical_system.light_wavelength, optical_system.numerical_aperture)
        
        normalization = np.sum(psf_on_grid_ij) * camera.dx^2
        psf_on_grid_ij = psf_on_grid_ij / normalization
    
        intermediate_img = fftshift(fft(ifftshift(psf_on_grid_ij)))
        
        for j in range(2*inference.padding_size+1):
            for i in range(2*inference.padding_size+1):
                mtf_on_grid_ij[i, j] = MTF_air_disk([f_corrected_grid_1D_ij[i], f_corrected_grid_1D_ij[j]], optical_system.light_wavelength, optical_system.numerical_aperture)
                if mtf_on_grid_ij[i, j] == 0.0:
                    intermediate_img[i, j] = 0.0 * intermediate_img[i, j]
	    
        FFT_point_spread_function_ij = ifftshift(intermediate_img) * camera.dx^2

    elif psf_type == "gaussian":
        for j in range(2*inference.padding_size+1):
            for i in range(2*inference.padding_size+1):
                x_e = [grid_physical_1D_ij[i], grid_physical_1D_ij[j]]
                psf_on_grid_ij[i, j] =  incoherent_PSF_gaussian([0.0, 0.0], x_e)
        
        FFT_point_spread_function_ij = fft(ifftshift(psf_on_grid_ij)) * camera.dx^2
        
    fft_plan = fft(psf_on_grid_ij)
    ifft_plan = ifft(psf_on_grid_ij)
    modulation_transfer_function_ij = np.abs(fftshift(FFT_point_spread_function_ij)) 
    modulation_transfer_function_ij_vectorized = modulation_transfer_function_ij.flatten()  / np.sum(modulation_transfer_function_ij)

    grid_physical_ij = 0
    normalization = 0
    psf_on_grid = 0
    mtf_on_grid = 0
    intermediate_img = 0
    f_corrected_grid_1D_ij = 0

    return fft_plan, ifft_plan, modulation_transfer_function_ij_vectorized

def get_widefield_image_ij(mean_img_ij, FFT_var, iFFT_var, img_ij, i, j, FFT_point_spread_function_ij, ifft_plan, inference):

    half_padding_size = inference.half_padding_size
    padding_size = inference.padding_size
    FFT_var = (FFT_var * FFT_point_spread_function_ij)
    iFFT_var = np.dot(ifft_plan, FFT_var)
    img_ij = fftshift(iFFT_var)
    mean_img_ij = np.abs(img_ij[half_padding_size:half_padding_size+padding_size, half_padding_size:half_padding_size+padding_size].real)
    
    return mean_img_ij

def log_Beta(a):
	z = sum(gammaln(a)) - gammaln(sum(a))
	return z

def log_Dirichlet(x, a):
	z = np.sum((a - 1.0) * math.log(x)) - log_Beta(a)
	return z

def log_Dirichlet_MTF(x, MTF_minus_one, log_Beta_MTF):
	x = MTF_minus_one * math.log(x)
	z = sum(x) - log_Beta_MTF
	return z

def get_log_likelihood_ij(i, j, mean_img_ij, shot_noise_img_ij, padding_size):
    
    log_likelihood = 0.0 
    for i in range(padding_size):
        for j in range(padding_size):
            log_likelihood += poisson.logpmf(mu=mean_img_ij[i, j] + np.linalg.eps, k=shot_noise_img_ij[i, j])
    
    return log_likelihood 

def get_log_prior_ij(FFT_var, img_ij, img_ij_abs, mod_fft_img_ij, i, j):
    
    fftshift(img_ij, FFT_var)
    img_ij_abs = np.abs(img_ij)
    mod_fft_img_ij = (img_ij_abs.flatten) / (sum(img_ij_abs)) + np.linalg.eps
    
    return log_Dirichlet_MTF(mod_fft_img_ij)

def bayesian_inference(modulation_transfer_function_ij_vectorized, conc_parameter):

    log_Beta_MTF = log_Beta(modulation_transfer_function_ij_vectorized + np.finfo(np.float64).eps)
    MTF_minus_one = (conc_parameter * modulation_transfer_function_ij_vectorized) - 1.0

def sample_object_neighborhood(temperature, object, shot_noise_image, mean_img_ij, proposed_mean_img_ij, FFT_var, 
                               iFFT_var, img_ij_abs, mod_fft_img_ij, n_accepted, inference, sub_img_size_x, fft_plan,
                               sub_img_size_y, padding_size, half_padding_size, covariance_object, raw_image_size_x, raw_image_size_y,
                               sub_raw_image, sub_gain_map, sub_offset_map, sub_error_map):
	
    for j in np.arange(padding_size, padding_size+sub_img_size_y):
        for i in np.arange(padding_size, padding_size+sub_img_size_x):

            old_value = object[i, j]
            shot_noise_img_ij = shot_noise_image[i - half_padding_size: i + half_padding_size+1, j - half_padding_size: j + half_padding_size+1]
            obj_ij = object[i - half_padding_size: i + half_padding_size+1, j - half_padding_size: j + half_padding_size+1] 
            
            img_ij = ifftshift(obj_ij)
            FFT_var = np.dot(fft_plan, img_ij)
            
            old_log_prior = get_log_prior_ij(FFT_var, img_ij, img_ij_abs, mod_fft_img_ij, i, j)  

 			# FFT_var gets modified in the following function
            
            mean_img_ij = get_widefield_image_ij(mean_img_ij, FFT_var, iFFT_var, img_ij, i, j)
            
            old_log_likelihood = get_log_likelihood_ij(i, j, mean_img_ij, shot_noise_img_ij)
            
            old_log_posterior = old_log_likelihood + old_log_prior
            
            old_jac = math.log(old_value)

            proposed_value = norm.rvs(loc=np.log(old_value), scale=np.sqrt(covariance_object), size=1)[0]

            print(proposed_value)
            
            proposed_value = np.exp(proposed_value)
            
            object[i, j] = proposed_value

            proposed_obj_ij = object[i - padding_size:i + padding_size+1, j - padding_size:j + padding_size+1] 
            img_ij = ifftshift(proposed_obj_ij)
            FFT_var = np.dot(fft_plan, img_ij)
            
            proposed_log_prior = get_log_prior_ij(FFT_var, img_ij, img_ij_abs, mod_fft_img_ij, i, j)
            
            proposed_mean_img_ij = get_widefield_image_ij(proposed_mean_img_ij, FFT_var, iFFT_var, img_ij, i, j)
            
            proposed_log_likelihood = get_log_likelihood_ij(i, j, proposed_mean_img_ij, shot_noise_img_ij)
            
            proposed_jac = np.log(proposed_value)
            proposed_log_posterior = proposed_log_likelihood + proposed_log_prior
            
            log_hastings = (1.0/temperature) * (proposed_log_posterior - old_log_posterior) + proposed_jac - old_jac
            log_rand = math.log(random.random())
            
            if log_hastings > log_rand:
                np.copyto(mean_img_ij, proposed_mean_img_ij)
                n_accepted += 1
            else:
                object[i, j] = old_value

 
			# Sample Intermediate Expected Photon Counts on each Pixel
			# Choose the central pixel in the mean image
			# for expected photon count
            
            expected_photon_count = mean_img_ij[half_padding_size, half_padding_size] 
            old_log_likelihood = norm.logpdf(sub_raw_image[i, j], loc=sub_gain_map[i, j]*expected_photon_count + sub_offset_map[i, j], scale=sub_error_map[i, j] + np.finfo(np.float64).eps)
            old_log_prior = poisson.logpmf(k = shot_noise_image[i, j], mu = expected_photon_count + np.finfo(np.float64).eps)
            
            old_log_posterior = old_log_likelihood  + old_log_prior
            
            proposed_shot_noise_pixel = poisson.rvs(mu = shot_noise_image[i, j], size=1)[0]
            
            new_log_likelihood = norm.logpdf(sub_raw_image[i, j], loc=sub_gain_map[i, j]*proposed_shot_noise_pixel + sub_offset_map[i, j], scale=sub_error_map[i, j] + np.finfo(np.float64).eps)
            
            new_log_prior = poisson.logpmf(k = proposed_shot_noise_pixel, mu = expected_photon_count + np.finfo(np.float64).eps)
 			
            new_log_posterior = new_log_likelihood  + new_log_prior

            log_forward_proposal_probability = poisson.logpmf(k = proposed_shot_noise_pixel, mu = shot_noise_image[i, j] + np.finfo(np.float64).eps)
            
            log_backward_proposal_probability = poisson.logpmf(k = shot_noise_image[i, j], mu = proposed_shot_noise_pixel + np.finfo(np.float64).eps)
            
            log_hastings = (1.0/temperature)* (new_log_posterior - old_log_posterior) + log_backward_proposal_probability - log_forward_proposal_probability
   
            log_rand = math.log(random.random())
            if log_hastings > log_rand: 
                shot_noise_image[i, j] = proposed_shot_noise_pixel
                
    return object, shot_noise_image, n_accepted

def sample_object_neighborhood_MLE(temperature, object, shot_noise_image, mean_img_ij, proposed_mean_img_ij, FFT_var,
                                   iFFT_var, img_ij, img_ij_abs, mod_fft_img_ij, n_accepted, inference, sub_img_size_x, fft_plan,
                                   sub_img_size_y, padding_size, half_padding_size, covariance_object, raw_image_size_x, raw_image_size_y,
                                   sub_raw_image, sub_gain_map, sub_offset_map, sub_error_map):
       
    for j in np.arange(padding_size, padding_size+sub_img_size_y):
        for i in np.arange(padding_size, padding_size+sub_img_size_x):  
                
            old_value = object[i, j]
            shot_noise_img_ij = shot_noise_image[i - half_padding_size: i + half_padding_size+1, j - half_padding_size: j + half_padding_size+1]
            obj_ij = object[i - half_padding_size: i + half_padding_size+1, j - half_padding_size: j + half_padding_size+1]
            img_ij = ifftshift(obj_ij)
            FFT_var = np.dot(fft_plan, img_ij)

 			# FFT_var gets modified in the following function
            mean_img_ij = get_widefield_image_ij(mean_img_ij, FFT_var, iFFT_var, img_ij, i, j)
            
            old_log_likelihood = get_log_likelihood_ij(i, j, mean_img_ij, shot_noise_img_ij)
            
            old_log_posterior = old_log_likelihood 
            old_jac = np.log(old_value)

            proposed_value = norm.rvs(loc=np.log(old_value), scale=np.sqrt(covariance_object), size=1)[0]
            proposed_value = np.exp(proposed_value)
            object[i, j] = proposed_value

            proposed_obj_ij = object[i - padding_size:i + padding_size+1, j - padding_size:j + padding_size+1]
            img_ij = ifftshift(proposed_obj_ij)
            FFT_var = np.dot(fft_plan, img_ij)

            proposed_mean_img_ij = get_widefield_image_ij(proposed_mean_img_ij, FFT_var, iFFT_var, img_ij, i, j)
    		
            proposed_log_likelihood = get_log_likelihood_ij(i, j, proposed_mean_img_ij, shot_noise_img_ij)
            
            proposed_jac = np.log(proposed_value)
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
            old_log_likelihood = norm.logpdf(sub_raw_image[i, j], loc=sub_gain_map[i, j]*expected_photon_count + sub_offset_map[i, j], scale=sub_error_map[i, j] + np.finfo(np.float64).eps)
            
            old_log_prior = poisson.logpmf(k = shot_noise_image[i, j], mu = expected_photon_count + np.finfo(np.float64).eps)
            
            old_log_posterior = old_log_likelihood  + old_log_prior
            proposed_shot_noise_pixel = poisson.rvs(mu = shot_noise_image[i, j], size=1)[0]
            
            new_log_likelihood = norm.logpdf(sub_raw_image[i, j], loc=sub_gain_map[i, j]*proposed_shot_noise_pixel + sub_offset_map[i, j], scale=sub_error_map[i, j] + np.finfo(np.float64).eps)

            new_log_prior = poisson.logpmf(k = proposed_shot_noise_pixel, mu = expected_photon_count + np.finfo(np.float64).eps)
            
            new_log_posterior = new_log_likelihood  + new_log_prior

            log_forward_proposal_probability = poisson.logpmf(k = proposed_shot_noise_pixel, mu = shot_noise_image[i, j] + np.finfo(np.float64).eps)

            log_backward_proposal_probability = poisson.logpmf(k = shot_noise_image[i, j], mu = proposed_shot_noise_pixel + np.finfo(np.float64).eps)

            log_hastings = (1.0/temperature) * (new_log_posterior - old_log_posterior) + log_backward_proposal_probability - log_forward_proposal_probability
            
            if log_hastings > 0.0:
                shot_noise_image[i, j] = proposed_shot_noise_pixel
    
    return object, shot_noise_image, n_accepted

def sampler(psf_typ0, abbe_diffraction_limit, physical_pixel_size, padding_size, median_photon_count, raw_img_size_x, raw_img_size_y, 
            sub_raw_img_size_x, sub_raw_img_size_y, n_procs_per_dim_x, n_procs_per_dim_y, total_draws, chain_burn_in_period, 
            chain_starting_temperature, chain_time_constant, annealing_starting_temperature, annealing_time_constant, annealing_frequency, 
            rng, input_raw_image, gain_map_with_padding, offset_map_with_padding, error_map_with_padding, modulation_transfer_function_ij_vectorized, 
            camera, optical_system, inference, im_raw, ip_raw, jm_raw, jp_raw):

	# Initialize
    draw = 1
    print("draw = ", draw)
    result_queue = multiprocessing.Queue()
	# Arrays for main variables of interest
    
    object = np.zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
    rng = np.random.default_rng()

    object[padding_size:-padding_size, padding_size:-padding_size] = rng.random((raw_img_size_x, raw_img_size_y)) 
    
    sum_object = np.zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
    mean_object = np.zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
    
    shot_noise_image = np.zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
    shot_noise_image[padding_size:-padding_size, padding_size:-padding_size] = np.ceil(np.abs(input_raw_image - np.median(offset_map_with_padding))).astype(np.int64)

    intermediate_img = np.zeros(3*raw_img_size_x, 3*raw_img_size_y)
    apply_reflective_BC_object(object, intermediate_img)
    apply_reflective_BC_shot(shot_noise_image, intermediate_img)
    
    mcmc_log_posterior = np.zeros(total_draws)
    mcmc_log_posterior[draw] = compute_full_log_posterior(object, shot_noise_image)
    
    averaging_counter = 0.0
 
	##@everywhere workers() begin
    n_accepted = 0
    temperature = 0.0
    
    sub_object = np.zeros(sub_raw_img_size_x+2*padding_size, sub_raw_img_size_y+2*padding_size)
    sub_shot_noise_img = np.zeros(sub_raw_img_size_x+2*padding_size, sub_raw_img_size_y+2*padding_size)

	# Arrays to store intermediate variables
    mean_img_ij = np.zeros(padding_size+1, padding_size+1)
    proposed_mean_img_ij = np.zeros(padding_size+1, padding_size+1)
    FFT_var = np.zeros(2*padding_size+1, 2*padding_size+1)
    iFFT_var = np.zeros(2*padding_size+1, 2*padding_size+1)
    img_ij = np.zeros(2*padding_size+1, 2*padding_size+1)
    img_ij_abs = np.zeros(2*padding_size+1, 2*padding_size+1)
    mod_fft_img_ij = np.zeros(np.size(modulation_transfer_function_ij_vectorized)[0]) 
	##end
       
    im = np.zeros(n_procs_per_dim_x)
    ip = np.zeros(n_procs_per_dim_x)
    jm = np.zeros(n_procs_per_dim_y)
    jp = np.zeros(n_procs_per_dim_y)

    for i in range(n_procs_per_dim_x):
        im[i+1] = padding_size + i*sub_raw_img_size_x + 1
        ip[i+1] = padding_size + (i+1)*sub_raw_img_size_x
        
    for j in range(n_procs_per_dim_y):
        jm[j+1] = padding_size + j*sub_raw_img_size_y + 1          
        jp[j+1] = padding_size + (j+1)*sub_raw_img_size_y


    n_accept = 0
    temperature = 0.0
    
    for draw in np.arange(2, total_draws+1):
        if draw > chain_burn_in_period:
            temperature = 1.0 + (annealing_starting_temperature-1.0)*np.exp(-((draw-chain_burn_in_period-1) % annealing_frequency)/annealing_time_constant)
        elif draw < chain_burn_in_period:
            temperature = 1.0 + (chain_starting_temperature-1.0)*np.exp(-((draw-1) % chain_burn_in_period)/chain_time_constant)
            
        print(draw)
        print(temperature)

		##@everywhere workers() begin
        sub_object = object[im_raw:ip_raw, jm_raw:jp_raw]
        sub_shot_noise_img = shot_noise_image[im_raw:ip_raw, jm_raw:jp_raw]
        
        i_procs = (5-1) % n_procs_per_dim_x # 并行计算的变量
        j_procs = (5-2 - i_procs) / n_procs_per_dim_x

        n_accepted = 0
        
        if i_procs + j_procs < draw+1:
            if draw > chain_burn_in_period:
                sub_object, sub_shot_noise_img, n_accepted = sample_object_neighborhood(temperature,  sub_object,  sub_shot_noise_img, mean_img_ij, proposed_mean_img_ij,
											FFT_var, iFFT_var, img_ij, img_ij_abs, mod_fft_img_ij, n_accepted)
            else:
                sub_object, sub_shot_noise_img, n_accepted = sample_object_neighborhood_MLE(temperature,  sub_object,  sub_shot_noise_img, mean_img_ij, proposed_mean_img_ij,
											FFT_var, iFFT_var, img_ij, img_ij_abs, mod_fft_img_ij, n_accepted)
        n_accept = 0
        for i in range(n_procs_per_dim_x):
            for j in range(n_procs_per_dim_y):
                
                procs_id = ((j-1)*n_procs_per_dim_x+(i-1)+2)
                sub_img = result_queue.get()#sub_img = view((@fetchfrom procs_id sub_object), :, :)
                
                object[im[i]:ip[i], jm[j]:jp[j]] = sub_img[padding_size:-padding_size, padding_size:padding_size]
                
                sub_img = result_queue.get()#sub_img = view((@fetchfrom procs_id sub_shot_noise_img), :, :)	
                shot_noise_image[im[i]:ip[i], jm[j]:jp[j]] = sub_img[padding_size:-padding_size, padding_size:-padding_size]
                
                accepted = 0 #accepted::Int64 = @fetchfrom procs_id n_accepted
                n_accept += accepted
        
        apply_reflective_BC_object(object, intermediate_img)
        apply_reflective_BC_shot(shot_noise_image, intermediate_img)
        
        print("accepted ", n_accept, " out of ", raw_img_size_x * raw_img_size_y, " pixels")	
        print("acceptance ratio = ", n_accept/ (raw_img_size_x * raw_img_size_y))	
        
        mcmc_log_posterior[draw] = compute_full_log_posterior(object, shot_noise_image)
        
        if (draw == chain_burn_in_period) or \
            ((draw > chain_burn_in_period) and 
            ((draw - chain_burn_in_period) % annealing_frequency > annealing_burn_in_period or 
            (draw - chain_burn_in_period) % annealing_frequency == 0) and 
            ((draw - chain_burn_in_period) % averaging_frequency == 0)):
            
            averaging_counter += 1.0
            sum_object = object
            mean_object = sum_object / averaging_counter
            
            print("Averaging Counter = ", averaging_counter)
            print("Saving Data...")
            
            save_data(draw, mcmc_log_posterior, object, shot_noise_image, mean_object, averaging_counter)
        
        if draw % plotting_frequency == 0:
       		plot_data(draw, object, mean_object, shot_noise_image, mcmc_log_posterior)
    


sampler()
rmprocs(workers())

if __name__ == "__main__":
    main()