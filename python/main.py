from math import ceil, sqrt
import numpy as np
import cv2
from numpy import pi
from psf import generate_psf
from multiprocessing import Pool
from time import time

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

    def get_camera_calibration_data(self, raw_image_size_x, raw_image_size_y):

        offset_map = self.offset * np.ones(raw_image_size_x, raw_image_size_y)
        gain_map = self.gain * np.ones(raw_image_size_x, raw_image_size_y)
        error_map = self.noise * np.ones(raw_image_size_x, raw_image_size_y)

        return offset_map, error_map, gain_map

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
    (input_raw_image, raw_image_size_x, raw_image_size_y, offset_map, error_map, gain_map, raw_image_with_padding, 
     gain_map_with_padding, offset_map_with_padding, error_map_with_padding, median_photon_count) = input_data(raw_image_path, camera)

    rng = np.random.default_rng()

    sub_raw_img_size_x = raw_image_size_x/inference.n_procs_per_dim_x 
    sub_raw_img_size_y = raw_image_size_y/inference.n_procs_per_dim_y

    modulation_transfer_function_vectorized = generate_psf(raw_image_size_x, raw_image_size_y, inference, optical_system, camera)

    testFL = np.arange(1,13)

    t1 = time()
    print('concurrent:') 
    pool = Pool(12)  # 创建拥有3个进程数量的进程池
    # testFL:要处理的数据列表，run：处理testFL列表中数据的函数
    pool.map(parallel, testFL)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    t2 = time()
    print("并行执行时间：", int(t2 - t1))

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

def input_data(raw_image_path, camera):

    input_raw_image = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
    input_raw_image = input_raw_image.astype(np.float64)    

    raw_image_size_x = input_raw_image.shape[0]
    raw_image_size_y = input_raw_image.shape[1]

    offset_map, error_map, gain_map = camera.get_camera_calibration_data(raw_image_size_x, raw_image_size_y)
    raw_image_with_padding = add_padding_reflective_BC(input_raw_image)
    gain_map_with_padding = add_padding_reflective_BC(gain_map)
    offset_map_with_padding = add_padding_reflective_BC(offset_map)
    error_map_with_padding = add_padding_reflective_BC(error_map)

    median_photon_count = np.ones((raw_image_size_x,raw_image_size_y))* np.median(abs((input_raw_image - np.ones((raw_image_size_x,raw_image_size_y))*np.median(offset_map_with_padding)) 
                                                                                       / (np.ones((raw_image_size_x,raw_image_size_y))*np.median(gain_map_with_padding))))
    
    return (input_raw_image, raw_image_size_x, raw_image_size_y, offset_map, error_map, gain_map, raw_image_with_padding, gain_map_with_padding,
            offset_map_with_padding, error_map_with_padding, median_photon_count)

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

def parallel():
    ...


if __name__ == "_main_":
    main()