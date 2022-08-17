%% CWSNN on PIMA dataset %% 
clear
close all
clc

%%% Read and process the dataset %%%
pima_indians_diabetes = load('pima_indians_diabetes.txt');
PIMA = pima_indians_diabetes(:,1:8);

%%% Parameter settings %%%
C = 2;                             % the number of class
SampleNum = size(PIMA,1);         % the number of samples
F = size(PIMA,2);                 % the number of features
G = 5;                             % the number of Gaussian receptive fields
beta = 1.5;                        % the parameter in the Gaussian codes
theta = 0.01;                      % the threthold of the Gaussian code
codingt = 100*ones(SampleNum,F*G); % the initial code t
ValidK = 10;
Times = 5;
AllResults = zeros(Times,5);
eta = 0.008;                        % the learning rate
theta1 = 2.5;                      % the threshold
D = 8;                             % the number of spiking layer
max_epoch = 100;
L = 25;                            % time interval
K = 6;                             % the number of synaptic
w = 0.5*rand(F*G,D,K);
tau = 7;

%%% code the inputs %%%%%%%%%%%%%%%%%%%%%
for n=1:F
    MaxValue = max(PIMA(:,n));
    MinValue = min(PIMA(:,n));
    for h=1:G
        center = MinValue+(2*h-3)/2*(MaxValue-MinValue)/(G-2);
        width = 1/beta*(MaxValue-MinValue)/(G-2);
        for j=1:SampleNum
            height = exp(-(PIMA(j,n)-center)^2/(width^2));
            if height>theta
                codingt(j,(n-1)*G+h) = floor((19+theta-20*height)/(2*(1-theta))+0.5);
            end
        end
    end
end

%%% code the outputs %%%
datay = pima_indians_diabetes(:,9);
datay1 = zeros(SampleNum,C);
for j = 1:SampleNum
    if datay(j) == 0
        datay1(j,:) = [1,0];
    else
        datay1(j,:) = [0,1];
    end
end
datapos = randperm(SampleNum);     % shuffle the order of samples
data = codingt(datapos,:);
datay = datay(datapos,:);
datay1 = datay1(datapos,:);

%%% Begin the main program including training and testing %%%
for time = 1:Times                  % each experiment 
    AllErr = zeros(ValidK,max_epoch); 
    AllTestErr = zeros(ValidK,max_epoch); 
    indices = crossvalind('Kfold',SampleNum,ValidK); 
    for validk = 1:ValidK           % each cross-validation
        testpos = (indices == validk); 
        trainpos = ~testpos;

        t = data(trainpos,:);       % the training dataset
        O1 = datay(trainpos); 
        O = datay1(trainpos,:); 

        testt=data(testpos,:);      % the testing dataset 
        testO1=datay(testpos);
        testO = datay1(testpos,:);
        testJ = size(testt,1);      % the number of test samples
        J = size(t,1);              % the number of training samples

        epoch = 1;
        epsilon = 0.02;

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

        ZR = zeros(J,C);               % the input real part of output layer
        ZI = zeros(J,C);               % the input image part of output layer
        fZI = zeros(J,C);
        y = zeros(J,C);

        alpha=ones(1,C);
        beta = ones(1,C);
        gamma = zeros(1,C);
        err_tmp = 1;
        TmpTestAcc = 0;
        TmpAcc = 0;

        testY = zeros(G*F,K);
        testY_D = zeros(G*F,K);
        testS = zeros(J,D);
        alltestY = zeros(J,G*F,K,D);
        testS_D = zeros(J,D);
        alltestY_D = zeros(J,G*F,K,D);

        testZR = zeros(testJ,C);      % the input real part of output layer
        testZI = zeros(testJ,C);      % the input image part of output layer
        testy = zeros(testJ,C);

        [Err,TestErr,Acc,TestAcc] = deal(zeros(max_epoch,1));
        %%%%%%% training process %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        while epoch<=max_epoch
        % while epoch<max_epoch && (Testaccuracy<0.98 || accuracy<0.98)
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
            for j = 1:J
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
            for j =1:J
                ZR(j,:) = NR(j,:)*vR-NI(j,:)*vI+v0R;
                ZI(j,:) = NI(j,:)*vR+NR(j,:)*vI+v0I;
                y(j,:) = 1./(1+exp(-(alpha.*ZR(j,:)+beta.*ZI(j,:)+gamma)));
            end

            %%%%%%%%%%%%%%%%%% caculate errors %%%%%%%%%%%
            Err(epoch) = sum(sum((y-O).*(y-O)))/(J*C);

            %%% the accuracy of the trainging %%%%%%%%%%%%%%%%%
            y1 = zeros(J,1);
            for j=1:J
                [mxy, y1(j)] = max(y(j,:));
                if (y1(j)-1)==O1(j)
                    Acc(epoch)=Acc(epoch)+1;
                end
            end
            Acc(epoch) = Acc(epoch)/J;

            %%%%%%%%%%%%%%%%%% update parameters %%%%%%%%%%%
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
                            w(m,h,k)=0;    
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
            for j = 1:testJ
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
            for j =1:testJ
                testZR(j,:) = testNR(j,:)*vR-testNI(j,:)*vI+v0R;
                testZI(j,:) = testNI(j,:)*vR+testNR(j,:)*vI+v0I;
                testy(j,:) = 1./(1+exp(-(alpha.*testZR(j,:)+beta.*testZI(j,:)+gamma)));
            end

             %%%%%%%%%%%%%%%%%% caculate errors %%%%%%%%%%%
            TestErr(epoch) = sum(sum((testy-testO).*(testy-testO)))/(testJ*C);

            %%% the accuracy of the testing %%%%%%%%%%%%%%%%%
            Testy1 = zeros(testJ,1);
            for j=1:testJ
                [mxy, Testy1(j)] = max(testy(j,:));
                if (Testy1(j)-1)==testO1(j)
                    TestAcc(epoch)=TestAcc(epoch)+1;
                end
            end
            TestAcc(epoch) = TestAcc(epoch)/testJ;

            if TestAcc(epoch)>=TmpTestAcc && Acc(epoch)>=TmpAcc
                TmpTestAcc = TestAcc(epoch);
                TmpAcc = Acc(epoch);
                TmpEpoch = epoch;
            end
            epoch=epoch+1;
        end
        % validk
        % close all
        % figure(1)
        % hold on
        % plot(1:max_epoch,Err)
        % plot(1:max_epoch,TestErr)
        % hold off
        % figure(2)
        % hold on
        % plot(1:max_epoch,Acc)
        % plot(1:max_epoch,TestAcc)
        % hold off
        % pause
        AllErr(validk,:)= Err';
        AllTestErr(validk,:)= TestErr';
        AllResults(time,:) = AllResults(time,:)+[TmpEpoch,Err(TmpEpoch),Acc(TmpEpoch),TestErr(TmpEpoch),TestAcc(TmpEpoch)];
    end
    
end
% figure(1)
% hold on
% plot(1:max_epoch,Err)
% plot(1:max_epoch,TestErr)
% hold off
% figure(2)
% hold on
% plot(1:max_epoch,Acc)
% plot(1:max_epoch,TestAcc)
% hold off
AllResults = AllResults./ValidK;
sum(AllResults,1)./Times

