import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import math
import multiprocessing
from scipy.special import j1
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.stats import dirichlet, poisson, norm

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
    def __init__(self, padding_size, half_padding_size, covariance_object, conc_parameter, n_procs_per_dim_x, n_procs_per_dim_y, total_draws, chain_burn_in_period, chain_starting_temperature, chain_time_constant, annealing_starting_temperature, annealing_time_constant, annealing_burn_in_period, annealing_frequency, averaging_frequency, plotting_frequency):
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
    sigma = math.sqrt(2.0)/(2.0*math.pi)*light_wavelength/numerical_aperture # Standard Deviation in PSF

    # Camera Parameters
    camera_pixel_size = 6.5 # In micrometers
    physical_pixel_size = camera_pixel_size/magnification #in micrometers
    dx = physical_pixel_size #physical grid spacing
    gain = 1.957
    offset = 100.0
    noise = 2.3
    noise_maps_available = False

    # Inference Parameters
    padding_size = 4*math.ceil(abbe_diffraction_limit/physical_pixel_size) 
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
    inference = Inference(padding_size, half_padding_size, covariance_object, conc_parameter, n_procs_per_dim_x, n_procs_per_dim_y, total_draws, chain_burn_in_period, chain_starting_temperature, chain_time_constant, annealing_starting_temperature, annealing_time_constant, annealing_burn_in_period, annealing_frequency, averaging_frequency, plotting_frequency)

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

    median_photon_count = np.ones((raw_image_size_x,raw_image_size_y))*np.median(abs((input_raw_image - np.ones((raw_image_size_x,raw_image_size_y))*np.median(offset_map_with_padding)) / (np.ones((raw_image_size_x,raw_image_size_y))*np.median(gain_map_with_padding))))
    return raw_image_size_x,raw_image_size_y

def get_camera_calibration_data(gain, offset, noise, raw_image_size_x, raw_image_size_y):

    offset_map = offset * np.ones(raw_image_size_x, raw_image_size_y)
    gain_map = gain * np.ones(raw_image_size_x, raw_image_size_y)
    error_map = noise * np.ones(raw_image_size_x, raw_image_size_y)

    return offset_map, error_map, gain_map

def add_padding_reflective_BC(input_img, padding_size):

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

def get_log_likelihood(object,shot_noise_image,inference,raw_image_with_padding,gain_map_with_padding,offset_map_with_padding,error_map_with_padding,raw_img_size_x,raw_img_size_y):
    log_likelihood = 0.0
    mean_image = get_mean_image(object)
    val_range_x = np.arange(inference.padding_size,inference.padding_size+raw_img_size_x)
    val_range_y = np.arange(inference.padding_size,inference.padding_size+raw_img_size_y)
    log_likelihood += np.sum(poisson.logpmf(shot_noise_image[val_range_x, val_range_y], mean_image[val_range_x, val_range_y] + np.finfo(np.float64).eps))
	
    norm((gain_map_with_padding[val_range_x, val_range_y]*shot_noise_image[val_range_x, val_range_y])+offset_map_with_padding[val_range_x, val_range_y],error_map_with_padding[val_range_x, val_range_y] + np.finfo(np.float64).eps),raw_image_with_padding[val_range_x, val_range_y]

    log_likelihood += sum(logpdf.(Normal.((gain_map_with_padding[val_range_x, val_range_y] .*shot_noise_image[val_range_x, val_range_y]) .+offset_map_with_padding[val_range_x, val_range_y],error_map_with_padding[val_range_x, val_range_y] .+ eps()),raw_image_with_padding[val_range_x, val_range_y]))
    log_likelihood += np.sum(norm.logpmf


	return log_likelihood
end


function get_log_likelihood(object::Matrix{Float64},
			shot_noise_image::Matrix{Float64})

	log_likelihood::Float64 = 0.0

	mean_image::Matrix{Float64} = get_mean_image(object)
	val_range_x = collect(padding_size+1:1:padding_size+raw_img_size_x)
	val_range_y = collect(padding_size+1:1:padding_size+raw_img_size_y)

	log_likelihood += sum(logpdf.(Poisson.(
				mean_image[val_range_x, val_range_y] .+ eps()),
				shot_noise_image[val_range_x, val_range_y]))
	log_likelihood += sum(logpdf.(Normal.(
                (gain_map_with_padding[val_range_x, val_range_y] .*
				shot_noise_image[val_range_x, val_range_y]) .+
					offset_map_with_padding[val_range_x, val_range_y],
                    error_map_with_padding[val_range_x, val_range_y] .+ eps()),
					raw_image_with_padding[val_range_x, val_range_y]))

	return log_likelihood
end

function compute_full_log_posterior(object::Matrix{Float64},
							shot_noise_image::Matrix{Float64})

	log_likelihood::Float64 = get_log_likelihood(object, shot_noise_image)
 	log_prior::Float64 = get_log_prior(object)
	log_posterior::Float64 = log_likelihood + log_prior

 	@show log_likelihood, log_prior, log_posterior

	return log_posterior
end


if __name__ == "__main__":
    main()