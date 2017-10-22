clear;
close all;

space_index = linspace(-1,1,100);
bell_curve1 = normpdf(space_index,0,0.3);
bell_curve2 = normpdf(space_index,0.5,0.3);
bell_curve3 = normpdf(space_index,-0.5,0.3);
%%%Spatio-temporal data with spatial component as gaussian, and peice wise
%%%constant  with sudden jump at time 50.



case1=  reshape(repmat(bell_curve1,1,100),[100,100]);
case2=  reshape(repmat(bell_curve2,1,100),[100,100]);
case3=  reshape(repmat(bell_curve3,1,100),[100,100]);
% case1=case1';
% case2=case2';
% case3=case3';
%%%%% Sine shaped temporal
sine_curve  = sin(linspace(-4*pi, 4*pi, 100));
sine_matrix = repmat(sine_curve',1,100);
sine_matrix = sine_matrix';%%%%%% correct %%%%%%%%%%%%%
%%% different scaling per class
case1 = case1 + 0.3* sine_matrix;
case2 = case2 + 0.6* sine_matrix;
case3 = case3 + 0.9* sine_matrix;
%%% temporal component add sudden jumps at time t =50 

case2(:,51:100) = case2(:,51:100)+0.1;
case3(:,51:100) = case3(:,51:100)-0.1;


%%%% Constructing tensor X by sample-by-space-time 
X = ones(90,100,100);

for i = 1:30
X(i,:,:)    = case1 + randn(100)*0.1;
X(i+30,:,:) = case2 + randn(100)*0.1;
X(i+60,:,:) = case3 + randn(100)*0.1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% CP decomposition 
CP_dec = sptensor(X);
     P = cp_als(CP_dec,3);  
%%% Tucker 3 decomposition
%T=tucker_als(CP_dec,[2 2 3]);
     T = tucker_als(CP_dec,3);
x1=P{1};
x2=P{2};
x3=P{3};
y1=T{1};
y2=T{2};
y3=T{3};

 figure()
 suptitle('CP Decomposition');
 subplot(3,3,1)
 plot(x1(:,1))
 title('u_1')
 subplot(3,3,2)
 plot(x1(:,2))
 title('u_2')
 subplot(3,3,3)
 plot(x1(:,3))
 title('u_3')
 subplot(3,3,4)
 plot(x2(:,1))
 title('v_1')
 subplot(3,3,5)
 plot(x2(:,2))
  title('v_2')
 subplot(3,3,6)
 plot(x2(:,3))
 title('v_3')
 subplot(3,3,7)
 plot(x3(:,1))
 title('w_1')
 subplot(3,3,8)
 plot(x3(:,2))
 title('w_2')
 subplot(3,3,9)
 plot(x3(:,3))
 title('w_3')
%%%
 figure()
 suptitle('Tucker Decomposition');
 subplot(3,3,1)
 plot(y1(:,1))
 subplot(3,3,2)
 plot(y1(:,2))
 subplot(3,3,3)
 plot(y1(:,3))
 subplot(3,3,4)
 plot(y2(:,1))
 subplot(3,3,5)
 plot(y2(:,2))
 subplot(3,3,6)
 plot(y2(:,3))
 subplot(3,3,7)
 plot(y3(:,1))
 subplot(3,3,8)
 plot(y3(:,2))
 subplot(3,3,9)
 plot(y3(:,3))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
