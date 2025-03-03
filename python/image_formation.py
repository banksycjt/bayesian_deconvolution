from numpy.fft import fft2, ifft2, ifftshift, fftshift
from numpy import dot, abs

def get_widefield_image(object, camera, FFT_point_spread_function):
    
    dx = camera.dx
    FFT_illuminated_object = fft2(ifftshift(object)) * dx^2
    FFT_final = FFT_point_spread_function * FFT_illuminated_object
    image = abs(fftshift(ifft2(FFT_final)).real)

    return image

def get_mean_image(object):

    final_image = get_widefield_image(object)

    return final_image

def get_widefield_image_ij(mean_img_ij, FFT_var, iFFT_var, img_ij, FFT_point_spread_function_ij, ifft_plan, inference):

    half_padding_size = inference.half_padding_size
    padding_size = inference.padding_size

    FFT_var = FFT_var * FFT_point_spread_function_ij
    iFFT_var = dot(ifft_plan, FFT_var)
    img_ij = fftshift(iFFT_var)
    mean_img_ij = abs(img_ij[half_padding_size:half_padding_size+padding_size, half_padding_size:half_padding_size+padding_size].real)
    
    return mean_img_ij
