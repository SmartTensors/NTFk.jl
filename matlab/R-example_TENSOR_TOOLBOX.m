space_index = linspace(-1,1,100);
bell_curve = normpdf(space_index,0,0.5);

%%%Spatio-temporal data with spatial component as gaussian, and peice wise
%%%constant  with sudden jump at time 50.

case1=  repmat(bell_curve',1,100);
case2=  repmat(bell_curve',1,100);
case3=  repmat(bell_curve',1,100);
case2(:,51:100) = case2(:,51:100)+0.1;
case3(:,51:100) = case3(:,51:100)-0.1;

%%%% Constructing tensor X by sample-by-space-time 
X=ones(90,100,100);

for i= 1:30
X(i,:,:)=case1 + reshape(normrnd(1,0.1,[1,10000]),[100,100]);
X(i+30,:,:)=case2 + reshape(normrnd(1,0.1,[1,10000]),[100,100]);
X(i+60,:,:)=case3 + reshape(normrnd(1,0.1,[1,10000]),[100,100]);
end

%Using Tensor Toolbox , CP-decomposition by als algorithm- alternating
%least squares algorithm.

Z=sptensor(X);
% P = parafac_als(X);
% R=tucker_als(X,[2 2 1]); 

P=cp_als(Z,3)                        