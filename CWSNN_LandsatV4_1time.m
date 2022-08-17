%% CWSNN on Landsat dataset %%
clear
close all
clc

%%%%%%%%%%%%%%%%% Data preprocessing %%%%%%%%%%%%%%%%%%%%%
%%% Read and process the dataset %%%
Landsat_data_numerical = load('sat.trn');
Jtrain = size(Landsat_data_numerical,1);       % the number of training samples
Landsat_data_numerical = Landsat_data_numerical(randperm(Jtrain),:); % shuffle the order of samples

datr = Landsat_data_numerical(:,1:36);         % Reduce 36 features to 4 features 
Landsatx=zeros(Jtrain,4);
for i_tr=1:size(datr,1)
    for k_tr=1:4
        a_tr=0;
        for j_tr=1:9
        a_tr=a_tr+datr(i_tr,4*j_tr-4+k_tr);
        end
        Landsatx(i_tr,k_tr)=a_tr/9;
    end
end
[Landsatx1,PS] = mapminmax(Landsatx',0,1);     % Normalization
Landsatx = Landsatx1';

Landsat_test_data_numerical = load('sat.tst');
Jtest = size(Landsat_test_data_numerical,1);   % the number of test samples
Landsat_test_data_numerical = Landsat_test_data_numerical(randperm(Jtest),:);
date=Landsat_test_data_numerical(:,1:36);
Landsat_tx =zeros(Jtest,4);
for i_te=1:size(date,1)
    for k_te=1:4
       a_te=0;
       for j_te=1:9
       a_te=a_te+date(i_te,4*j_te-4+k_te);
       end
       Landsat_tx(i_te,k_te)=a_te/9;
    end
end
Landsat_tx1 = mapminmax('apply',Landsat_tx',PS); % Normalization
Landsat_tx = Landsat_tx1';

data = [Landsatx,Landsat_data_numerical(:,37);Landsat_tx,Landsat_test_data_numerical(:,37)];
SampleNum = size(data,1);                        % the number of samples

%%% parameter settings %%%
F = size(data,2)-1;                              % the number of features
eta = 0.01;                                      % the learning rate
G = 6;                                           % the number of Gaussian receptive fields
D = 23;                                          % the number of spiking layer
C = 6;                                           % the number of class
K = 7;                                           % the number of synaptic
L = 25;                                          % time interval
max_epoch = 500;    
theta1 = 2.5;                                    % the threshold
epoch = 1;
w = abs(0.22+0.15*randn(F*G,D,K));
tau = 7;
epsilon = 0.01;     
beta = 1.5;                                      % the parameter or the Gaussian code
theta = 0.01;                                    % the threthold of the Gaussian code
Jbatch = 128;                                    % The number of samples for each batch of data

%%% Code inputs %%%
codingData = 100*ones(SampleNum,F*G); 
for n=1:F
    MaxValue = max(data(:,n));
    MinValue = min(data(:,n));
    for h=1:G
        center = MinValue+(2*h-3)/2*(MaxValue-MinValue)/(G-2);
        width = 1/beta*(MaxValue-MinValue)/(G-2);
        for j=1:SampleNum 
            height = exp(-(data(j,n)-center)^2/(width^2));
            if height>theta
                codingData(j,(n-1)*G+h) = floor((19+theta-20*height)/(2*(1-theta))+0.5);
            end
        end
    end
end

%%% Code outputs %%%
dataO1 = data(:,F+1);  
dataO = zeros(SampleNum,C);
for j = 1:SampleNum
    if dataO1(j) == 1
        dataO(j,:) = [1,0,0,0,0,0];
    elseif dataO1(j) == 2
        dataO(j,:) = [0,1,0,0,0,0];
    elseif dataO1(j) == 3
        dataO(j,:) = [0,0,1,0,0,0];
    elseif dataO1(j) == 4
        dataO(j,:) = [0,0,0,1,0,0];
    elseif dataO1(j) == 5
        dataO(j,:) = [0,0,0,0,1,0];
    else
        dataO1(j) = 6;
        dataO(j,:) = [0,0,0,0,0,1];
    end
end

%%% Divide training set and test set %%%
trainpos = 1:4435;
testpos = 4436:6435;
codingDataTrain = codingData(trainpos,:);   % the training set
dataOTrain = dataO(trainpos,:);
dataO1Train = dataO1(trainpos,:);
testx = codingData(testpos,:);              % the testing set
testO1 = dataO1(testpos,:);
testO = dataO(testpos,:);

%%% Divide batch %%%
BatchNum = ceil(Jtrain/Jbatch);
pos = cell(BatchNum,1);
for i = 1:BatchNum
    if i<BatchNum
        pos{i} = 1+Jbatch*(i-1):Jbatch*i;
    else
       pos{i} =  1+Jbatch*(i-1):Jtrain;
    end
end

%%% Pre-allocated variable space %%%
vR = 2*rand(D,C)-1;
vI = 2*rand(D,C)-1;
v0R = ones(1,C);
v0I = ones(1,C);
alpha=ones(1,C);
beta = ones(1,C);
gamma = zeros(1,C);
Testerr = zeros(max_epoch,1);
TestACC = zeros(max_epoch,1);
Y = zeros(G*F,K);
Y_D = zeros(G*F,K);
err_tmp = 1;
accuracy = 0;
[err,ACC] = deal(zeros(max_epoch,BatchNum));
Testaccuracy1 = 0;
testY = zeros(G*F,K);
testY_D = zeros(G*F,K);
testS = zeros(Jtest,D);
testS_D = zeros(Jtest,D);
testZR = zeros(Jtest,C);                   % the input real part of output layer
testZI = zeros(Jtest,C);                   % the input image part of output layer
testy = zeros(Jtest,C);
testNR = zeros(Jtest,D);
testNI = zeros(Jtest,D);

%%%%%%% training process %%%%%%%%%%%%%%%%%%%%%%%%%%%%
while epoch<=max_epoch
    for order=1:BatchNum
        x = codingDataTrain(pos{order},:);
        O1 = dataO1Train(pos{order},:);
        O = dataOTrain(pos{order},:);
        Jbatch = size(O,1);
        T = 100.*ones(Jbatch,D);
        NR = zeros(Jbatch,D);
        NI = zeros(Jbatch,D);
        
        S = zeros(Jbatch,D);
        allY = zeros(Jbatch,G*F,K,D);
        S_D = zeros(Jbatch,D);
        allY_D = zeros(Jbatch,G*F,K,D);

        ZR = zeros(Jbatch,C);               % the input real part of output layer
        ZI = zeros(Jbatch,C);               % the input image part of output layer
        fZI = zeros(Jbatch,C);
        y = zeros(Jbatch,C);
        
        %%% Forward propagation %%%
        for j = 1:Jbatch
            %%% the spiking part %%%%%%%%%%%%%%%%
            for l = 1:1:L
                for m = 1:G*F
                    for k = 1:K
                        dk = k;
                        tmp = l-x(j,m)-dk;
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
                            S_D(j,h) = sum(sum(w1(:,:,h).*Y_D,2));
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
        for j = 1:Jbatch
            muT = mean(nonzeros(T(j,:)));
            muS_D = mean(nonzeros(S_D(j,:)));
            sigmaTsquare = 0;
            sigmaSsquare = 0;
            numT = 0;
            for h = 1:D
                if T(j,h)~=0
                    numT = numT + 1;
                    sigmaTsquare = sigmaTsquare+(T(j,h)-muT)^2;
                    sigmaSsquare = sigmaSsquare+(S_D(j,h)-muS_D)^2;
                end
            end                        
            sigmaTsquare = sigmaTsquare/numT;
            sigmaSsquare = sigmaSsquare/numT;
            for h = 1:D
                if T(j,h)~=0
                    NR(j,h) = (T(j,h)-muT)./sqrt(sigmaTsquare+epsilon);
                    NI(j,h) = (S_D(j,h)-muS_D)./sqrt(sigmaSsquare+epsilon);
                end 
            end
        end
        
        %%%%%% output part %%%%%%%%%%%
        for j =1:Jbatch
            ZR(j,:) = NR(j,:)*vR-NI(j,:)*vI+v0R;
            ZI(j,:) = NI(j,:)*vR+NR(j,:)*vI+v0I;
            y(j,:) = 1./(1+exp(-(alpha.*ZR(j,:)+beta.*ZI(j,:)+gamma)));
        end
 
        %%%%%%%%%%%%%%%%%% caculate errors %%%%%%%%%%%
        err(epoch,order) = sum(sum((y-O).*(y-O)))/(Jbatch*C);
        err_tmp = err(epoch,order);
        %%% the accuracy of the trainging %%%%%%%%%%%%%%%%%
        y1 = zeros(Jbatch,1);
        accuracy = 0;
        for j=1:Jbatch
            [mxy, y1(j)] = max(y(j,:));
            if y1(j)==O1(j)
                accuracy=accuracy+1;
            end
        end
        accuracy = accuracy/Jbatch;
        ACC(epoch,order) = accuracy;
    
 
        %%%%%%%% Backpropagation %%%%%%%%%%%
        %%% reinitialize parameters %%%%%%%%%%%
        Delta_vR = zeros(D,C);
        Delta_vI = zeros(D,C);
        Delta_w = zeros(F*G,D,K);
        Derivative_NR = 1/sqrt(sigmaTsquare+epsilon);
        Derivative_NI = 1/sqrt(sigmaSsquare+epsilon);
 
        DerivativeFixed = (y-O).*y.*(1-y); % Fixed
        Delta_alpha = sum(DerivativeFixed.*ZR,1);
        Delta_beta = sum(DerivativeFixed.*ZI,1);
        Delta_gamma = sum(DerivativeFixed,1);
        Delta_v0R = sum(DerivativeFixed,1).*alpha;
        Delta_v0I = sum(DerivativeFixed,1).*beta;
        for j=1:Jbatch
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
    end

    %%%%%%% testing process %%%%%%%%%%%%%%%%%%%%
    testT = 100.*ones(Jtest,D);
    for j = 1:Jtest
        %%% the spiking part %%%%%%%%%%%%%%%%
        for l = 1:1:L
            for m = 1:G*F
                for k = 1:K
                    dk = k;
                    tmp = l-testx(j,m)-dk;
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
    for j = 1:Jtest
        muT = mean(nonzeros(testT(j,:)));
        muS_D = mean(nonzeros(testS_D(j,:)));
        numT = 0;
        sigmaTsquare = 0;
        sigmaSsquare = 0;
        for h = 1:D
            if testT(j,h)~=0
                numT = numT + 1;
                sigmaTsquare = sigmaTsquare+(testT(j,h)-muT)^2;
                sigmaSsquare = sigmaSsquare+(testS_D(j,h)-muS_D)^2;
            end
        end
        sigmaTsquare = sigmaTsquare/numT;
        sigmaSsquare = sigmaSsquare/numT;
        for h = 1:D
            if testT(j,h)~=0
                testNR(j,h) = (testT(j,h)-muT)/sqrt(sigmaTsquare+epsilon);
                testNI(j,h) = (testS_D(j,h)-muS_D)/sqrt(sigmaSsquare+epsilon);
            end 
        end
    end
     %%%%%% output part %%%%%%%%%%%
    for j =1:Jtest
        testZR(j,:) = testNR(j,:)*vR-testNI(j,:)*vI+v0R;
        testZI(j,:) = testNI(j,:)*vR+testNR(j,:)*vI+v0I;
        testy(j,:) = 1./(1+exp(-(alpha.*testZR(j,:)+beta.*testZI(j,:)+gamma)));
    end

     %%%%%%%%%%%%%%%%%% caculate errors %%%%%%%%%%%
    Testerr(epoch) = sum(sum((testy-testO).*(testy-testO)))/(Jtest*C);

    %%% the accuracy of the testing %%%%%%%%%%%%%%%%%
    Testy1 = zeros(Jtest,1);
    Testaccuracy = 0;
    for j=1:Jtest
        [mxy, Testy1(j)] = max(testy(j,:));
        if Testy1(j)==testO1(j)
            Testaccuracy=Testaccuracy+1;
        end
    end
    Testaccuracy = Testaccuracy/Jtest;
    if Testaccuracy>Testaccuracy1
        output=[epoch,sum(err(epoch,:))/BatchNum,sum(ACC(epoch,:))/BatchNum,Testerr(epoch),Testaccuracy];
        Testaccuracy1 = Testaccuracy;
    end
    TestACC(epoch) = Testaccuracy;

    [epoch,sum(err(epoch,:))/BatchNum,sum(ACC(epoch,:))/BatchNum,Testerr(epoch),TestACC(epoch)];
    
    epoch=epoch+1;

    postrain = randperm(Jtrain);
    codingDataTrain = codingDataTrain(postrain,:);   % the training set
    dataOTrain = dataOTrain(postrain,:);
    dataO1Train = dataO1Train(postrain,:);
end
close all
figure(1)
hold on
plot(1:max_epoch,sum(err,2)/BatchNum)
plot(1:max_epoch,Testerr(1:epoch-1))
hold off
figure(2)
hold on
plot(1:max_epoch,sum(ACC,2)/BatchNum)
plot(1:max_epoch,TestACC(1:epoch-1))
hold off
output                                                % the best results 