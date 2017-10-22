%%%%%%%%% Instantaneous mixing data;c
clear
close all
rng(2017)
a = rand(1,20);
b = rand(1,20);
W = [a' b'];
H = [.1,1,0,0,0.1; 0,0,0.1,0.5,0.2];
X = W * H;
tsize  = [20,5,4];
T_orig = rand([20,5,4]);
T_orig(:,:,1) = X;
T_orig(:,:,2) = X*2;
T_orig(:,:,3) = X*3;
T_orig(:,:,4) = X*4;

%%% CP decomposition
CP_dec = sptensor(T_orig);
C      = cp_als(CP_dec, 2);
N      = cp_nmu(CP_dec, 2);
%%% Tucker 3 decomposition
T      = tucker_als(CP_dec, [2,2,1]);

x1 = C{1};
x2 = C{2};
x3 = C{3};
n1 = N{1};
n2 = N{2};
n3 = N{3};
y1 = T{1};
y2 = T{2};
y3 = T{3};

figure()
suptitle('CP Decomposition');
subplot(3,3,1)
hold all
plot(x1(:,1))
plot(a)
title('u_1')
hold all
subplot(3,3,2)
hold all
plot(x1(:,2))
plot(b)
title('u_2')

subplot(3,3,4)
hold all
plot(x2(:,1))
plot(H(1,:))
title('v_1')
subplot(3,3,5)
hold all
plot(x2(:,2))
plot(H(2,:))
title('v_2')

subplot(3,3,7)
hold all
plot(x3(:,1))
plot([1,2,3,4])
title('w_1')
subplot(3,3,8)
hold all
plot(x3(:,2))
plot([1,2,3,4])
title('w_2')

figure()
suptitle('NMU CP Decomposition');
subplot(3,3,1)
hold all
plot(n1(:,1))
plot(a)
title('u_1')
hold all
subplot(3,3,2)
hold all
plot(n1(:,2))
plot(b)
title('u_2')

subplot(3,3,4)
hold all
plot(n2(:,1))
plot(H(1,:))
title('v_1')
subplot(3,3,5)
hold all
plot(n2(:,2))
plot(H(2,:))
title('v_2')

subplot(3,3,7)
hold all
plot(n3(:,1))
plot([1,2,3,4])
title('w_1')
subplot(3,3,8)
hold all
plot(n3(:,2))
plot([1,2,3,4])
title('w_2')

%%%
figure()
suptitle('Tucker Decomposition')
subplot(3,3,1)
hold all
plot(y1(:,1))
plot(a)
title('u_1')
hold all
subplot(3,3,2)
hold all
plot(y1(:,2))
plot(b)
title('u_2')

subplot(3,3,4)
hold all
plot(y2(:,1))
plot(H(1,:))
title('v_1')
subplot(3,3,5)
hold all
plot(y2(:,2))
plot(H(2,:))
title('v_2')

subplot(3,3,7)
hold all
plot(y3(:,1))
plot([1,2,3,4])
title('w_1')
