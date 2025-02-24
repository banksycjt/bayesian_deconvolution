import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import math

def main():
    ...

def input_parameter():
    
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

def input_data():

    input_raw_image = get_input_raw_image()
    raw_image_size_x = input_raw_image.shape[0]
    raw_image_size_y = input_raw_image.shape[1]
    gain = 1.957
    offset = 100.0
    noise = 2.3
    offset_map, error_map, gain_map = get_camera_calibration_data(gain, offset, noise, raw_image_size_x, raw_image_size_y)

def generate_psf():
    ...

def get_input_raw_image():
    
    file_name = ""+"raw_image.tif"
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float64)
    return img

def get_camera_calibration_data(gain, offset, noise, raw_img_size_x, raw_img_size_y):

    offset_map = offset * np.ones(raw_img_size_x, raw_img_size_y)
    gain_map = gain * np.ones(raw_img_size_x, raw_img_size_y)
    error_map = noise * np.ones(raw_img_size_x, raw_img_size_y)

    return offset_map, error_map, gain_map

def add_padding_reflective_BC(input_img):


	size_x = np.size(input_img)[0]
	size_y = np.size(input_img)[1]

	img = np.zeros(3*size_x, 3*size_y)
	img[size_x+1:end-size_x,size_y+1:end-size_y] = input_img 
	img[1:size_x,
			size_y+1:end-size_y] .= input_img[end:-1:1, :] 
	img[end-size_x+1:end,
			size_y+1:end-size_y] .= input_img[end:-1:1, :] 
	img[size_x+1:end-size_x,
			1:size_y] .= input_img[:, end:-1:1] 
	img[size_x+1:end - size_x,
        end-size_y+1:end] .= input_img[:, end:-1:1] 
	img[1:size_x,1:size_y] .= input_img[end:-1:1, end:-1:1] 
	img[1:size_x,end-size_y+1:end] .= input_img[end:-1:1, end:-1:1] 
	img[end-size_x+1:end,1:size_y] .= input_img[end:-1:1, end:-1:1] 
	img[end-size_x+1:end,end-size_y+1:end] .= input_img[end:-1:1, end:-1:1] 

	return img[size_x-padding_size+1:2*size_x+padding_size, size_y-padding_size+1:2*size_y+padding_size]
end