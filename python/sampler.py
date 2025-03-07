from numpy import zeros, random, ceil, abs, median, int64, size, arange, exp
from reflective_BC import apply_reflective_BC_object, apply_reflective_BC_shot
from global_posterior import compute_full_log_posterior
from multiprocessing import Pool

def sampler(psf_type, abbe_diffraction_limit, physical_pixel_size, padding_size, median_photon_count, raw_img_size_x, raw_img_size_y, 
            sub_raw_img_size_x, sub_raw_img_size_y, n_procs_per_dim_x, n_procs_per_dim_y, chain_burn_in_period, 
            chain_starting_temperature, chain_time_constant, annealing_starting_temperature, annealing_time_constant, annealing_frequency, 
            input_raw_image, raw_image_with_padding, gain_map_with_padding, offset_map_with_padding, error_map_with_padding, 
            modulation_transfer_function_ij_vectorized, camera, optical_system, inference, im_raw, ip_raw, jm_raw, jp_raw, 
            modulation_transfer_function_vectorized, raw_image_size_x, raw_image_size_y):

	# Initialize
    draw = 0
    print("draw = ", draw)
    total_draws = inference.total_draws

	# Arrays for main variables of interest
    
    object = zeros((raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size))
    rng = random.default_rng()

    object[padding_size:-padding_size, padding_size:-padding_size] = rng.random((raw_img_size_x, raw_img_size_y)) 
    
    sum_object = zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
    mean_object = zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
    
    shot_noise_image = zeros(raw_img_size_x+2*padding_size, raw_img_size_y+2*padding_size)
    shot_noise_image[padding_size:-padding_size, padding_size:-padding_size] = ceil(abs(input_raw_image - median(offset_map_with_padding))).astype(int64)

    intermediate_img = zeros(3*raw_img_size_x, 3*raw_img_size_y)
    object = apply_reflective_BC_object(object, intermediate_img, padding_size)
    shot_noise_image = apply_reflective_BC_shot(shot_noise_image, intermediate_img, padding_size)
    
    mcmc_log_posterior = zeros(total_draws)
    mcmc_log_posterior[draw] = compute_full_log_posterior(object, shot_noise_image, inference, raw_image_with_padding, gain_map_with_padding, 
                               offset_map_with_padding, error_map_with_padding, raw_img_size_x,raw_img_size_y, 
                               modulation_transfer_function_vectorized, raw_image_size_x, raw_image_size_y)
    
    averaging_counter = 0.0
 
	##@everywhere workers() begin
    n_accepted = 0
    temperature = 0.0
    
    sub_object = zeros(sub_raw_img_size_x+2*padding_size, sub_raw_img_size_y+2*padding_size)
    sub_shot_noise_img = zeros(sub_raw_img_size_x+2*padding_size, sub_raw_img_size_y+2*padding_size)

	# Arrays to store intermediate variables
    mean_img_ij = zeros(padding_size+1, padding_size+1)
    proposed_mean_img_ij = zeros(padding_size+1, padding_size+1)
    FFT_var = zeros(2*padding_size+1, 2*padding_size+1)
    iFFT_var = zeros(2*padding_size+1, 2*padding_size+1)
    img_ij = zeros(2*padding_size+1, 2*padding_size+1)
    img_ij_abs = zeros(2*padding_size+1, 2*padding_size+1)
    mod_fft_img_ij = zeros(size(modulation_transfer_function_ij_vectorized)) 
	##end
       
    im = zeros(n_procs_per_dim_x)
    ip = zeros(n_procs_per_dim_x)
    jm = zeros(n_procs_per_dim_y)
    jp = zeros(n_procs_per_dim_y)

    for i in range(n_procs_per_dim_x):
        im[i+1] = padding_size + i*sub_raw_img_size_x + 1
        ip[i+1] = padding_size + (i+1)*sub_raw_img_size_x
        
    for j in range(n_procs_per_dim_y):
        jm[j+1] = padding_size + j*sub_raw_img_size_y + 1          
        jp[j+1] = padding_size + (j+1)*sub_raw_img_size_y


    n_accept = 0
    temperature = 0.0
    
    for draw in arange(2, total_draws+1):
        if draw > chain_burn_in_period:
            temperature = 1.0 + (annealing_starting_temperature-1.0)*exp(-((draw-chain_burn_in_period-1) % annealing_frequency)
                                                                         /annealing_time_constant)
        elif draw < chain_burn_in_period:
            temperature = 1.0 + (chain_starting_temperature-1.0)*exp(-((draw-1) % chain_burn_in_period)/chain_time_constant)
            
        print(draw)
        print(temperature)

        data_list = []
    for i in range(12):
        data = (i+1, raw_image_size_x, raw_image_size_y, sub_raw_img_size_x, sub_raw_img_size_y, inference, optical_system, camera, 
                raw_image_with_padding, gain_map_with_padding, offset_map_with_padding, error_map_with_padding)
        data_list.append(data)
        

    print('concurrent:') 
    pool = Pool(12)  # 创建拥有12个进程数量的进程池
    # testFL:要处理的数据列表，run：处理testFL列表中数据的函数
    pool.map(parallel, arg = data_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
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
        ##end
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
 
def sub_sampler()