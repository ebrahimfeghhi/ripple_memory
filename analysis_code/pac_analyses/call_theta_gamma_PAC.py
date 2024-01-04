from theta_gamma_PAC import * 

metric = 'MOVI'

skip_modes = [0, 1, 3]
#encoding_mode = 0 # 0 for recalled data, for 1 for encoding data
#if encoding_mode == 0:
#    skip_modes.extend([0,1])

if int(sys.argv[1]) not in skip_modes:
    if metric == 'MOVI':
        save_MOVI(mode=int(sys.argv[1]))
    if metric == 'MI':
        save_MOVI(mode=int(sys.argv[1]))