%% CWSNN on Sunspot dataset %%
clear
close all
clc

%%% Read dataset %%%
Sunspot = csvread('E:\Documents\Dataset\Sunspot_ms1834_2001.csv',0,3); 
[data,PS]=mapminmax(Sunspot',0,1);              % Normalization
figure(1)
plot(1001:length(data),data(1001:2000))         % Draw a graph of the original data

%%% Construct training set and testing set %%%
F = 5;                                          % Time series window length
BreakPoint = 1000;                              % Take 221 as the demarcation point, 1700-1920 as the training set, 1921-1979 as the test set
J = BreakPoint-F;                               % Number of training set samples
testJ = length(data)-BreakPoint;                % Number of testing set samples

[train_x,train_y,test_x,test_y] = deal(zeros(J,F),zeros(J,1),zeros(testJ,F),zeros(testJ,1));
for j = 1:J
    train_x(j,:) = data(j:j+F-1);
    train_y(j) = data(j+F);
end
for j = 1:testJ
    test_x(j,:) = data(J+j:J+j+F-1);
    test_y(j) = data(J+j+F);
end

SampleNum = size(train_x,1);              % the number of samples
G = 5;                                    % the number of Gaussian receptive fields
beta = 1.5;                               % the parameter or the Gaussian code
theta = 0.01;                             % the threthold of the Gaussian code
t = 100*ones(J,F*G);                      % the initial code t
testt = 100*ones(testJ,F*G);              % the initial code t

%%%%% coding process %%%%%%%%%%%%%%%%%%%%%
for n=1:F
    MaxValue = max(max(train_x(:,n)),max(test_x(:,n)));
    MinValue = min(min(train_x(:,n)),min(test_x(:,n)));
    for h=1:G
        center = MinValue+(2*h-3)/2*(MaxValue-MinValue)/(G-2);
        width = 1/beta*(MaxValue-MinValue)/(G-2);
        for j=1:J
            height = exp(-(train_x(j,n)-center)^2/(width^2));
            if height>theta
                t(j,(n-1)*G+h) = floor((19+theta-20*height)/(2*(1-theta))+0.5);
            end
        end
        for j=1:testJ
            height = exp(-(test_x(j,n)-center)^2/(width^2));
            if height>theta
                testt(j,(n-1)*G+h) = floor((19+theta-20*height)/(2*(1-theta))+0.5);
            end
        end
    end
end

%%% parameter settings %%%
eta = 0.01;                            % the learning rate
theta1 = 2.5;                          % the threshold
D = 7;                                 % the number of spiking layer
max_epoch = 10000;
epoch = 1;
MSE = zeros(max_epoch,1);
TestMSE = zeros(max_epoch,1);
[TrainRMSE,TrainMAE,TestRMSE,TestMAE] = deal(zeros(max_epoch,1));
L = 25;                                % time interval
K = 7;                                 % the number of synaptic
w = 0.5*rand(F*G,D,K);
tau = 7;

epsilon = 0.03;
C = 1;                                 % the number of class
vR = 2*rand(D,C)-1;
vI = 2*rand(D,C)-1;
v0R = ones(1,C);
v0I = ones(1,C);

Y = zeros(G*F,K);
Y_D = zeros(G*F,K);
S = zeros(J,D);
allY = zeros(J,G*F,K,D);
S_D = zeros(J,D);
allY_D = zeros(J,G*F,K,D);

ZR = zeros(J,C);                      % the input real part of output layer
ZI = zeros(J,C);                      % the input image part of output layer
fZI = zeros(J,C);
y = zeros(J,C);

alpha=ones(1,C);
beta = ones(1,C);
gamma = zeros(1,C);
err_tmp = 1;
Testaccuracy = 0;
Testaccuracy1 = 0;
accuracy = 0;

testY = zeros(G*F,K);
testY_D = zeros(G*F,K);
testS = zeros(J,D);
alltestY = zeros(J,G*F,K,D);
testS_D = zeros(J,D);
alltestY_D = zeros(J,G*F,K,D);

testZR = zeros(testJ,C);              % the input real part of output layer
testZI = zeros(testJ,C);              % the input image part of output layer
testy = zeros(testJ,C);
TestMAEtmp = 1;

%%%%%%% training process %%%%%%%%%%%%%%%%%%%%%%%%%%%%
while epoch<=max_epoch
    T = 100.*ones(J,D);
    NR = zeros(J,D);
    NI = zeros(J,D);
    for j = 1:J
        %%% the spiking part %%%%%%%%%%%%%%%%
        for l = 1:1:L
            for m = 1:G*F
                for k = 1:K
                    dk = k;
                    tmp = l-t(j,m)-dk;
                    if tmp>0
                        Y(m,k) = tmp/tau*exp(1-tmp/tau);
                        Y_D(m,k) = (1-tmp/tau)/tau*exp(1-tmp/tau);
                    else
                        Y(m,k) = 0;
                        Y_D(m,k) = 0;
                    end
                end
            end
            for h = 1:D
                if T(j,h)==100
                    w1 = permute(w,[1,3,2]);
                    S(j,h) = sum(sum(w1(:,:,h).*Y));
                    if S(j,h)>theta1
                        T(j,h) = l;
                        allY(j,:,:,h) = Y;
                        S_D(j,h) = sum(sum(w1(:,:,h).*Y_D));
                        allY_D(j,:,:,h) = Y_D;
                    end
                end
            end
            if max(T(j,:))<100
                break
            end
        end
    end
    %%%%%% normalization batch part %%%%%%%%   
    muT = mean(nonzeros(T));
    muS_D = mean(nonzeros(S_D));
    numT = 0;
    sigmaTsquare = 0;
    sigmaSsquare = 0;
    for j = 1:J
        for h = 1:D
            if T(j,h)~=0
                numT = numT + 1;
                sigmaTsquare = sigmaTsquare+(T(j,h)-muT)^2;
                sigmaSsquare = sigmaSsquare+(S_D(j,h)-muS_D)^2;
            end
        end
    end
    sigmaTsquare = sigmaTsquare/numT;
    sigmaSsquare = sigmaSsquare/numT;
    for j = 1:J
        for h = 1:D
            if T(j,h)~=0
                NR(j,h) = (T(j,h)-muT)./sqrt(sigmaTsquare+epsilon);
                NI(j,h) = (S_D(j,h)-muS_D)./sqrt(sigmaSsquare+epsilon);
            end 
        end
    end

     %%%%%% output part %%%%%%%%%%%
    for j =1:J
        ZR(j,:) = NR(j,:)*vR-NI(j,:)*vI+v0R;
        ZI(j,:) = NI(j,:)*vR+NR(j,:)*vI+v0I;
        y(j,:) = 1./(1+exp(-(alpha.*ZR(j,:)+beta.*ZI(j,:)+gamma)));
    end

     %%%%%%%%%%%%%%%%%% caculate errors %%%%%%%%%%%
    MSE(epoch) = sum(sum((y-train_y).*(y-train_y)))/(J*C);
    TrainRMSE(epoch) = sqrt(sum(sum((y-train_y).*(y-train_y)))/(J*C));
    TrainMAE(epoch) = sum(abs(y-train_y))/(J*C);

  

    %%%%%%%%%%%%%%%%%% update parameters %%%%%%%%%%%
    %%% reinitialize parameters %%%%%%%%%%%
    Delta_vR = zeros(D,C);
    Delta_vI = zeros(D,C);
    Delta_w = zeros(F*G,D,K);
    Derivative_NR = 1/sqrt(sigmaTsquare+epsilon);
    Derivative_NI = 1/sqrt(sigmaSsquare+epsilon);
    
    DerivativeFixed = (y-train_y).*y.*(1-y); % Fixed

    Delta_alpha = sum(DerivativeFixed.*ZR,1);
    Delta_beta = sum(DerivativeFixed.*ZI,1);
    Delta_gamma = sum(DerivativeFixed,1);
    Delta_v0R = sum(DerivativeFixed,1).*alpha;
    Delta_v0I = sum(DerivativeFixed,1).*beta;
    for j=1:J
        for h=1:D
            for p=1:C
                Delta_vR(h,p) = Delta_vR(h,p)+DerivativeFixed(j,p)*(alpha(p)*NR(j,h)+beta(p)*NI(j,h));
                Delta_vI(h,p) = Delta_vI(h,p)+DerivativeFixed(j,p)*(-alpha(p)*NI(j,h)+beta(p)*NR(j,h));
            end
            if S_D(j,h)~=0
                for p=1:C
                    for m=1:G*F
                        for k=1:K
                            Delta_w(m,h,k) = Delta_w(m,h,k)+DerivativeFixed(j,p)*(-(alpha(p)*vR(h,p)+beta(p)*vI(h,p))*Derivative_NR...
                                /S_D(j,h)*allY(j,m,k,h)+(-alpha(p)*vI(h,p)+beta(p)*vR(h,p))*Derivative_NI*allY_D(j,m,k,h));
                        end
                    end
                end
            end
        end
    end

    %%% update %%%%%%%%%%%%%%%%%%%
    alpha = alpha-eta*Delta_alpha;
    beta = beta-eta*Delta_beta;
    gamma = gamma-eta*Delta_gamma;
    v0R = v0R-eta*Delta_v0R;
    v0I = v0I-eta*Delta_v0I;
    vR = vR-eta.*Delta_vR;
    vI = vI-eta.*Delta_vI;
    w = w-eta.*Delta_w;
    for m=1:F*G
        for h=1:D
            for k=1:K
                if w(m,h,k)<0
                    w(m,h,k)=0;    % if w<0, set w=0
                end
            end
        end
    end
             

    %%%%% testing process %%%%%%%%%%%%%%%%%%%%
    testT = 100.*ones(testJ,D);
    testNR = zeros(testJ,D);
    testNI = zeros(testJ,D);
    for j = 1:testJ
        %%% the spiking part %%%%%%%%%%%%%%%%
        for l = 1:1:L
            for m = 1:G*F
                for k = 1:K
                    dk = k;
                    tmp = l-testt(j,m)-dk;
                    if tmp>0
                        testY(m,k) = tmp/tau*exp(1-tmp/tau);
                        testY_D(m,k) = (1-tmp/tau)/tau*exp(1-tmp/tau);
                    else
                        testY(m,k) = 0;
                        testY_D(m,k) = 0;
                    end
                end
            end
            for h = 1:D
                if testT(j,h)==100
                    w1 = permute(w,[1,3,2]);
                    testS(j,h) = sum(sum(w1(:,:,h).*testY));
                    if testS(j,h)>theta1
                        testT(j,h) = l;
                        testS_D(j,h) = sum(sum(w1(:,:,h).*testY_D));
                    end
                end
            end
            if max(testT(j,:))<100
                break
            end
        end
    end
    %%%%%% normalization batch part %%%%%%%%   
    muT = mean(nonzeros(testT));
    muS_D = mean(nonzeros(testS_D));
    numT = 0;
    sigmaTsquare = 0;
    sigmaSsquare = 0;
    
    for j = 1:testJ
        for h = 1:D
            if testT(j,h)~=0
                numT = numT + 1;
                sigmaTsquare = sigmaTsquare+(testT(j,h)-muT)^2;
                sigmaSsquare = sigmaSsquare+(testS_D(j,h)-muS_D)^2;
            end
        end
    end
    sigmaTsquare = sigmaTsquare/numT;
    sigmaSsquare = sigmaSsquare/numT;
    
    for j = 1:testJ
        for h = 1:D
            if testT(j,h)~=0
                testNR(j,h) = (testT(j,h)-muT)/sqrt(sigmaTsquare+epsilon);
                testNI(j,h) = (testS_D(j,h)-muS_D)/sqrt(sigmaSsquare+epsilon);
            end 
        end
    end
     %%%%%% output part %%%%%%%%%%%
    for j =1:testJ
        testZR(j,:) = testNR(j,:)*vR-testNI(j,:)*vI+v0R;
        testZI(j,:) = testNI(j,:)*vR+testNR(j,:)*vI+v0I;
        testy(j,:) = 1./(1+exp(-(alpha.*testZR(j,:)+beta.*testZI(j,:)+gamma)));
    end

     %%%%%%%%%%%%%%%%%% caculate errors %%%%%%%%%%%
     TestMSE(epoch) = sum(sum((testy-test_y).*(testy-test_y)))/(testJ*C);
     TestRMSE(epoch) = sqrt(sum(sum((testy-test_y).*(testy-test_y)))/(testJ*C));
     TestMAE(epoch) = sum(abs(testy-test_y))/(testJ*C);
     if TestMAE(epoch)<TestMAEtmp
         TestMAEtmp = TestMAE(epoch);
         testytmp = testy;
     end
    
  
    epoch=epoch+1;
end
figure(2)
hold on
plot(1:(epoch-1),MSE(1:epoch-1))
plot(1:(epoch-1),TestMSE(1:epoch-1))
hold off
[MSE(epoch-1),TestMSE(epoch-1),TrainRMSE(epoch-1),TestRMSE(epoch-1),TestMAE(epoch-1),TestMAE(epoch-1)]  % the final results
[MSE(epoch-1),TrainRMSE(epoch-1),TestRMSE(epoch-1)]