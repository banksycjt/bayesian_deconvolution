import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import math
from scipy.special import j1

def generate_psf(raw_image_size_x, raw_image_size_y, inference, optical_system, camera):

    dx = camera.dx
    padding_size = inference.padding_size
    psf_type = optical_system.psf_type
    
    grid_physical_1D_x = dx * np.arange(-(raw_image_size_x/2 + padding_size),(raw_image_size_x/2 + padding_size)) # in micrometers
    grid_physical_1D_y = dx * np.arange(-(raw_image_size_y/2 + padding_size),(raw_image_size_y/2 + padding_size)) # in micrometers
    
    grid_physical_length_x = (raw_image_size_x + 2*padding_size - 1)*dx			
    grid_physical_length_y = (raw_image_size_y + 2*padding_size - 1)*dx

    df_x = 1/(grid_physical_length_x) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 
    df_y = 1/(grid_physical_length_y) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 

    f_corrected_grid_1D_x = df_x * np.arange(-(raw_image_size_x/2 + padding_size),(raw_image_size_x/2 + padding_size)) # in units of micrometer^-1
    f_corrected_grid_1D_y = df_y * np.arange(-(raw_image_size_y/2 + padding_size),(raw_image_size_y/2 + padding_size)) # in units of micrometer^-1

    mtf_on_grid = np.zeros(raw_image_size_x+2*inference.padding_size, raw_image_size_y+2*inference.padding_size)
    psf_on_grid = np.zeros(raw_image_size_x+2*inference.padding_size, raw_image_size_y+2*inference.padding_size)

    if psf_type == "airy_disk":
        
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

    elif psf_type == "gaussian":

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