## well done
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from math import exp, sqrt, pi, acos
from scipy.special import j1
from numpy.linalg import norm
from numpy import arange, zeros, sum, array, shape

def generate_psf(raw_image_size_x, raw_image_size_y, inference, optical_system, camera):

    dx = camera.dx
    padding_size = inference.padding_size
    psf_type = optical_system.psf_type
    light_wavelength = optical_system.light_wavelength
    numerical_aperture = optical_system.numerical_aperture
    sigma = optical_system.sigma
    
    grid_physical_1D_x = dx * arange(-(raw_image_size_x/2 + padding_size),(raw_image_size_x/2 + padding_size)) # in micrometers
    grid_physical_1D_y = dx * arange(-(raw_image_size_y/2 + padding_size),(raw_image_size_y/2 + padding_size)) # in micrometers
    
    grid_physical_length_x = (raw_image_size_x + 2*padding_size - 1)*dx			
    grid_physical_length_y = (raw_image_size_y + 2*padding_size - 1)*dx

    df_x = 1/(grid_physical_length_x) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 
    df_y = 1/(grid_physical_length_y) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 

    f_corrected_grid_1D_x = df_x * arange(-(raw_image_size_x/2 + padding_size),(raw_image_size_x/2 + padding_size)) # in units of micrometer^-1
    f_corrected_grid_1D_y = df_y * arange(-(raw_image_size_y/2 + padding_size),(raw_image_size_y/2 + padding_size)) # in units of micrometer^-1

    mtf_on_grid = zeros((raw_image_size_x+2*padding_size, raw_image_size_y+2*padding_size))
    psf_on_grid = zeros((raw_image_size_x+2*padding_size, raw_image_size_y+2*padding_size))

    if psf_type == "airy_disk":
        
        for j in range(raw_image_size_y + 2*padding_size):
            for i in range(raw_image_size_x + 2*padding_size):
                x_e = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
                psf_on_grid[i, j] =  incoherent_PSF_airy_disk([0.0, 0.0], x_e, light_wavelength, numerical_aperture)
 		
        normalization = sum(psf_on_grid) * dx^2
        psf_on_grid = psf_on_grid / normalization
        intermediate_img = fftshift(fft2(ifftshift(psf_on_grid)))

        for j in range(raw_image_size_y + 2*padding_size):
            for i in range(raw_image_size_x + 2*padding_size):
                mtf_on_grid[i, j] = MTF_air_disk([f_corrected_grid_1D_x[i], f_corrected_grid_1D_y[j]], light_wavelength, numerical_aperture)
                if mtf_on_grid[i, j] == 0.0:
                    intermediate_img[i, j] = 0.0 * intermediate_img[i, j]

        FFT_point_spread_function = ifftshift(intermediate_img) 

    elif psf_type == "gaussian":

        for j in range(raw_image_size_y+ 2*padding_size):
            for i in range(raw_image_size_x + 2*padding_size):
                x_e = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
                psf_on_grid[i, j] =  incoherent_PSF_gaussian([0.0, 0.0], x_e, sigma)
        
        FFT_point_spread_function = fft2(ifftshift(psf_on_grid))
        
        
    modulation_transfer_function = abs(fftshift(FFT_point_spread_function))[padding_size:-padding_size, padding_size:-padding_size] 
    modulation_transfer_function_vectorized = (modulation_transfer_function.flatten()) / sum(modulation_transfer_function)
    
    return modulation_transfer_function_vectorized

def incoherent_PSF_airy_disk(x_c, x_e, light_wavelength, numerical_aperture):
    x_c = array(x_c)
    x_e = array(x_e)
    f_number = 1/(2*numerical_aperture) ##approx
    return (jinc(norm(x_c - x_e)/(light_wavelength*f_number)))**2

def incoherent_PSF_gaussian(x_c, x_e, sigma):
    x_c = array(x_c)
    x_e = array(x_e)
    return exp(-norm(x_c-x_e)**2/(2.0*sigma**2)) /(sqrt(2.0*pi) * sigma)**(shape(x_e)[0])

def jinc(x):
    if x == 0:
        return 0.5 
    else:
        return j1(x) / x

def MTF_air_disk(f_vector, light_wavelength, numerical_aperture):
		f_number = 1/(2*numerical_aperture) ##approx
		highest_transmitted_frequency = 1.0 / (light_wavelength*f_number) 
		norm_f = norm(f_vector) / highest_transmitted_frequency
		if norm_f < 1.0:
			mtf = 2.0/pi * (acos(norm_f) - norm_f*sqrt(1 - norm_f^2))
		elif norm_f >= 1.0:
			mtf = 0.0
		return mtf