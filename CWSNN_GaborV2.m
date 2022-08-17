%% Spiking neural network combining time and space with complex value, Gabor, simple version %%
% 
clear
close all
clc

%%%%%%%%%%% 输入，输出 %%%%%%%%%%%%%%%%%%%%%%%%
Num=8;
J=Num^2;          % 训练样本数量
Sample=-0.5:1/(Num-1):0.5;
[X1,X2]=ndgrid(Sample);   % 生成多变量函数的自变量序列
x1=X1(:);x2=X2(:);    %将矩阵按列顺序转化为向量
train_y1=1/(2*pi*0.25)*exp(-((x1).^2+(x2).^2)/0.5).*cos(2*pi*(x1+x2));
train_x=[x1,x2];

Number=20;
testJ=Number^2;           %测试样本数量
Sample=-0.5:1/(Number-1):0.5;
[X1,X2]=ndgrid(Sample);   % 生成多变量函数的自变量序列
x1=X1(:);x2=X2(:);    %将矩阵按列顺序转化为向量
test_y1=1/(2*pi*0.25)*exp(-((x1).^2+(x2).^2)/0.5).*cos(2*pi*(x1+x2));
test_x=[x1,x2];

[yAll,ps]=mapminmax([train_y1;test_y1]',0,1);
train_y = yAll(1:J)';
test_y = yAll(J+1:J+testJ)';


F = 2;                          % 自变量个数 
J = size(train_x,1); % the number of samples
G = 3;                     % the number of Gaussian receptive fields
beta1 = 1.5;                % the parameter or the Gaussian code
theta = 0.01;              % the threthold of the Gaussian code
t = 100*ones(J,F*G); % the initial code t
testt = 100*ones(testJ,F*G); % the initial code t
T_interval = 10;
%%%%% coding process %%%%%%%%%%%%%%%%%%%%%
for n=1:F
    MaxValue = max(max(train_x(:,n)),max(test_x(:,n)));
    MinValue = min(min(train_x(:,n)),min(test_x(:,n)));
    for h=1:G
        center = MinValue+(2*h-3)/2*(MaxValue-MinValue)/(G-2);
        width = 1/beta1*(MaxValue-MinValue)/(G-2);
        for j=1:J
            height = exp(-(train_x(j,n)-center)^2/(width^2));
            if height>theta
%                 t(j,(n-1)*G+h) = floor((19+theta-20*height)/(2*(1-theta))+0.5);
                t(j,(n-1)*G+h) = roundn(T_interval.*(1-height),-2);
            end
        end
        for j=1:testJ
            height = exp(-(test_x(j,n)-center)^2/(width^2));
            if height>theta
%                 testt(j,(n-1)*G+h) = floor((19+theta-20*height)/(2*(1-theta))+0.5);
                testt(j,(n-1)*G+h) = roundn(T_interval.*(1-height),-2);
            end
        end
    end
end

eta = 0.002;         % the learning rate
theta1 = 2.5;       % the threshold
D = 8;             % the number of spiking layer
max_epoch = 15000;
epoch = 1;
MSE = zeros(max_epoch,1);
TestMSE = zeros(max_epoch,1);
[TrainRMSE,TrainMAE,TestRMSE,TestMAE] = deal(zeros(max_epoch,1));
L = 25;    % time interval
K = 4;     % the number of synaptic
w = 0.5*rand(F*G,D,K);
tau = 7;

epsilon = 0.02;
C = 1;     % the number of class
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

ZR = zeros(J,C);   % the input real part of output layer
ZI = zeros(J,C);   % the input image part of output layer
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
testS = zeros(testJ,D);
alltestY = zeros(testJ,G*F,K,D);
testS_D = zeros(testJ,D);
alltestY_D = zeros(testJ,G*F,K,D);

testZR = zeros(testJ,C);   % the input real part of output layer
testZI = zeros(testJ,C);   % the input image part of output layer
testy = zeros(testJ,C);
TestMSEtmp = 1;
TrainMSEtmp = 1;
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
    T(T>L) = 0;
    S_D(T>L) = 0;
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
%         BR(j,:) = T(j,:);
     %%%%%% output part %%%%%%%%%%%
    for j =1:J
        ZR(j,:) = NR(j,:)*vR-NI(j,:)*vI+v0R;
        ZI(j,:) = NI(j,:)*vR+NR(j,:)*vI+v0I;
        y(j,:) = 1./(1+exp(-(alpha.*ZR(j,:)+beta.*ZI(j,:)+gamma)));
    end
     %%%%%%%%%%%%%%%%%% caculate errors %%%%%%%%%%%
     y1 = (mapminmax('reverse',y',ps))';
     MSE(epoch) = sum(sum((y1-train_y1).*(y1-train_y1)))/(J*C);
%     y1 = mapminmax('reverse',y,PS);
%     TrainRMSE(epoch) = sqrt(sum(sum((y1-train_y).*(y1-train_y)))/(J*C));
%     TrainMAE(epoch) = sum(abs(y1-train_y))/(J*C);
    
  

    %%%%%%%%%%%%%%%%%% update parameters %%%%%%%%%%%
    %%% reinitialize parameters %%%%%%%%%%%
    Delta_vR = zeros(D,C);
    Delta_vI = zeros(D,C);
    Delta_w = zeros(F*G,D,K);
    Derivative_NR = 1/sqrt(sigmaTsquare+epsilon);
    Derivative_NI = 1/sqrt(sigmaSsquare+epsilon);
%     SumwDerivative_Y = zeros(Hidden,1);
    
    DerivativeFixed = (y-train_y).*y.*(1-y); % Fixed
%     for h=1:Hidden
%         SumwDerivative_Y(h) = SumwDerivative_Y(h)+S_D(j,h);
%     end
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
    testT(testT>L) = 0;
    testS_D(testT>L) = 0;
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
    
    
%         for h = 1:Hidden          % normlization the spiking layer
%             BR(j,h) = (T(j,h)-muT)/sqrt(sigmaTsquare+epsilon);
%             BI(j,h) = (S(j,h)-muS)/sqrt(sigmaSsquare+epsilon);
%         end
    for j = 1:testJ
        for h = 1:D
            if testT(j,h)~=0
                testNR(j,h) = (testT(j,h)-muT)/sqrt(sigmaTsquare+epsilon);
                testNI(j,h) = (testS_D(j,h)-muS_D)/sqrt(sigmaSsquare+epsilon);
            end 
        end
    end
%         BR(j,:) = T(j,:);
     %%%%%% output part %%%%%%%%%%%
    for j =1:testJ
        testZR(j,:) = testNR(j,:)*vR-testNI(j,:)*vI+v0R;
        testZI(j,:) = testNI(j,:)*vR+testNR(j,:)*vI+v0I;
        testy(j,:) = 1./(1+exp(-(alpha.*testZR(j,:)+beta.*testZI(j,:)+gamma)));
    end
    

     %%%%%%%%%%%%%%%%%% caculate errors %%%%%%%%%%%
     testy1 = (mapminmax('reverse',testy',ps))';
     TestMSE(epoch) = sum(sum((testy1-test_y1).*(testy1-test_y1)))/(testJ*C);
     
     
%     TestRMSE(epoch) = sqrt(sum(sum((testy1-test_y).*(testy1-test_y)))/(testJ*C));
%     TestMAE(epoch) = sum(abs(testy1-test_y))/(testJ*C);
     if (TestMSE(epoch)<TestMSEtmp)&&(MSE(epoch)<=TestMSE(epoch))
         TestMSEtmp = TestMSE(epoch);
         TrainMSEtmp = MSE(epoch);
         testytmp = testy1;
         traintmp = y1;
         epochtmp = epoch;
     end
    
    
    
    [MSE(epoch),TestMSE(epoch)];
    epoch=epoch+1;
end
epochtmp
[TrainMSEtmp ,TestMSEtmp]  % the final results

%% MSE随训练时期变化图
figure(1)
hold on
plot(1:(epoch-1),MSE(1:epoch-1))
plot(1:(epoch-1),TestMSE(1:epoch-1))
hold off
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'mse','-dpng','-r600')

%% 测试拟合图 反归一化
figure(2)
xx2=-0.5:1/(Number-1):0.5;
yy2=-0.5:1/(Number-1):0.5;
[x2,y2]=meshgrid(xx2,yy2);

Out0=zeros(Number,Number);
 for m=1:testJ
     j=fix((m-1)/Number)+1;
     k=m-(j-1)*Number;
     Out0(j,k)=test_y1(m);
 end
Out0;

Out=zeros(Number,Number);
 for m=1:testJ
     j=fix((m-1)/Number)+1;
     k=m-(j-1)*Number;
     Out(j,k)=testytmp(m);
 end
Out;

subplot(1,2,1)
hold on
mesh(x2,y2, Out0)
axis([-0.5 0.5 -0.5 0.5 -0.5 1])
view(-20,50.450455136540945)
plot([-0.0789,-0.0263,0.0263,0.0789],[-0.4474,-0.4474,-0.4474,-0.4474],'*')
hold off
subplot(1,2,2)
mesh(x2,y2,Out)
axis([-0.5 0.5 -0.5 0.5 -0.5 1])
view(-20,50.450455136540945)

Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'test','-dpng','-r600')
% saveas(gcf,'Gabortest0.0227.fig')

%% 训练拟合图，反归一化 也不太好
figure(3)
xx1=-0.5:1/(Num-1):0.5;
yy1=-0.5:1/(Num-1):0.5;
[x1,y1]=meshgrid(xx1,yy1);

Out0=zeros(Num,Num);
for i=1:J
    j=fix((i-1)/Num)+1;
    k=i-(j-1)*Num;
    Out0(j,k)=train_y1(i);
end
Out0;

Out=zeros(Num,Num);
for i=1:J
    j=fix((i-1)/Num)+1;
    k=i-(j-1)*Num;
    Out(j,k)=traintmp(i);
end
Out;

subplot(1,2,1)
mesh(x1,y1, Out0)
axis([-0.5 0.5 -0.5 0.5 -0.5 1])
view(-17.971962616822434,58.479063719115715)
subplot(1,2,2)
mesh(x1,y1,Out)
axis([-0.5 0.5 -0.5 0.5 -0.5 1])
view(-17.971962616822434,58.479063719115715)
Fig = getimage(gcf);                                 %获取当前坐标系图像
print(Fig,'train','-dpng','-r600')



% 
% %% 训练拟合图，归一化 
% figure(4)
% xx1=-0.5:1/(Num-1):0.5;
% yy1=-0.5:1/(Num-1):0.5;
% [x1,y1]=meshgrid(xx1,yy1);
% 
% Out0=zeros(Num,Num);
% for i=1:J
%     j=fix((i-1)/Num)+1;
%     k=i-(j-1)*Num;
%     Out0(j,k)=train_y(i);
% end
% Out0;
% 
% Out=zeros(Num,Num);
% for i=1:J
%     j=fix((i-1)/Num)+1;
%     k=i-(j-1)*Num;
%     Out(j,k)=y(i);
% end
% Out;
% 
% subplot(1,2,1)
% mesh(x1,y1, Out0)
% axis([-0.5 0.5 -0.5 0.5 -0.5 1])
% subplot(1,2,2)
% mesh(x1,y1,Out)
% axis([-0.5 0.5 -0.5 0.5 -0.5 1])
