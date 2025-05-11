function get_widefield_image_ij!(mean_img_ij::Matrix{Float64}, FFT_var::Matrix{ComplexF64}, 
	iFFT_var::Matrix{ComplexF64}, img_ij::Matrix{ComplexF64}, i::Int64, j::Int64)

	FFT_var .= (FFT_var .* FFT_point_spread_function_ij) # fft of img_ij dot fft of psf
	mul!(iFFT_var, ifft_plan, FFT_var) # img_ij convolves with psf
   	fftshift!(img_ij, iFFT_var)
	mean_img_ij .= abs.(real.(view(img_ij, half_padding_size+1:half_padding_size+1+padding_size,
		half_padding_size+1:half_padding_size+1+padding_size)))

  	return mean_img_ij
end
