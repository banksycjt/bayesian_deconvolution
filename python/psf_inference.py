from numpy import arange, zeros, sum, abs
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from psf import incoherent_PSF_airy_disk, MTF_air_disk, incoherent_PSF_gaussian

def psf_for_inference(camera, optical_system, inference):

    dx = camera.dx
    padding_size = inference.paddingsize
    psf_type = optical_system.psf_type
    light_wavelength = optical_system.light_wavelength
    numerical_aperture = optical_system.numerical_aperture

    grid_physical_1D_ij = dx * arange(-padding_size, padding_size+1) # in micrometers
    grid_physical_length_ij = 2.0*padding_size*dx

    df_ij = 1/(grid_physical_length_ij) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 
    f_corrected_grid_1D_ij = df_ij * arange(-padding_size, padding_size+1)

    mtf_on_grid_ij = zeros((2*padding_size+1, 2*padding_size+1))
    psf_on_grid_ij = zeros((2*padding_size+1, 2*padding_size+1))

    if psf_type == "airy_disk":
        for j in range(2*padding_size+1):
               for i in range(2*padding_size+1):
                     x_e = [grid_physical_1D_ij[i], grid_physical_1D_ij[j]]
                     psf_on_grid_ij[i, j] =  incoherent_PSF_airy_disk([0.0, 0.0], x_e, light_wavelength, numerical_aperture)
        
        normalization = sum(psf_on_grid_ij) * dx^2
        psf_on_grid_ij = psf_on_grid_ij / normalization
    
        intermediate_img = fftshift(fft2(ifftshift(psf_on_grid_ij)))
        
        for j in range(2*padding_size+1):
            for i in range(2*padding_size+1):
                x_e = [f_corrected_grid_1D_ij[i], f_corrected_grid_1D_ij[j]]
                mtf_on_grid_ij[i, j] = MTF_air_disk(x_e, light_wavelength, numerical_aperture)
                if mtf_on_grid_ij[i, j] == 0.0:
                    intermediate_img[i, j] = 0.0 * intermediate_img[i, j]
	    
        FFT_point_spread_function_ij = ifftshift(intermediate_img) * dx^2

    elif psf_type == "gaussian":
        for j in range(2*padding_size+1):
            for i in range(2*padding_size+1):
                x_e = [grid_physical_1D_ij[i], grid_physical_1D_ij[j]]
                psf_on_grid_ij[i, j] =  incoherent_PSF_gaussian([0.0, 0.0], x_e)
        
        FFT_point_spread_function_ij = fft2(ifftshift(psf_on_grid_ij)) * dx^2
        
    fft_plan = fft2(psf_on_grid_ij)
    ifft_plan = ifft2(psf_on_grid_ij)
    modulation_transfer_function_ij = abs(fftshift(FFT_point_spread_function_ij)) 
    modulation_transfer_function_ij_vectorized = modulation_transfer_function_ij.flatten()  / sum(modulation_transfer_function_ij)

    return fft_plan, ifft_plan, modulation_transfer_function_ij_vectorized
