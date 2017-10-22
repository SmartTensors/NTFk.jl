%%%%%%%%% Instantaneous mixing data;c
clear
close all
rng(2017)
tsize = [2, 10, 5]
T = h5read('dNTF/contamination/tensor.jld', '/T')

%%% CP decomposition
CP_dec = sptensor(T_orig);
C      = cp_als(CP_dec, 5);
N      = cp_nmu(CP_dec, 5);
%%% Tucker 3 decomposition
T      = tucker_als(CP_dec, [1,5,5]);

TC = double(C)
TN = double(N)
TT = double(T)

figure()
suptitle('Original');
subplot(5,1,1)
image(T_orig(:,:,1))
title('t_1')
subplot(5,1,2)
image(T_orig(:,:,2))
title('t_2')
subplot(5,1,3)
image(T_orig(:,:,3))
title('t_3')
subplot(5,1,4)
image(T_orig(:,:,4))
title('t_4')
subplot(5,1,5)
image(T_orig(:,:,5))
title('t_5')

figure()
suptitle('NCP');
subplot(5,1,1)
image(TN(:,:,1))
title('t_1')
subplot(5,1,2)
image(TN(:,:,2))
title('t_2')
subplot(5,1,3)
image(TN(:,:,3))
title('t_3')
subplot(5,1,4)
image(TN(:,:,4))
title('t_4')
subplot(5,1,5)
image(TN(:,:,5))
title('t_5')

figure()
suptitle('CP');
subplot(5,1,1)
image(TC(:,:,1))
title('t_1')
subplot(5,1,2)
image(TC(:,:,2))
title('t_2')
subplot(5,1,3)
image(TC(:,:,3))
title('t_3')
subplot(5,1,4)
image(TC(:,:,4))
title('t_4')
subplot(5,1,5)
image(TC(:,:,5))
title('t_5')

figure()
suptitle('Tucker');
subplot(5,1,1)
image(TT(:,:,1))
title('t_1')
subplot(5,1,2)
image(TT(:,:,2))
title('t_2')
subplot(5,1,3)
image(TT(:,:,3))
title('t_3')
subplot(5,1,4)
image(TT(:,:,4))
title('t_4')
subplot(5,1,5)
image(TT(:,:,5))
title('t_5')