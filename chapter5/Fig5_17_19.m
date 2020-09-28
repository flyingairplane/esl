% [Lo_D,Hi_D] = wfilters('sym8','d');
% [a,w]=ScaleWaveFig(Lo_D,3);

%%%%%%%绘制基函数张成的空间，每个尺度均可张成2^j个函数
% [wp,x] = wpfun('haar',20,10);%20为图的数量，10表示有2^10个格子
% figure;
% plot(wp(size(wp,1),:),'-*');
% for i=1:size(wp,1)
%     subplot(size(wp,1),1,i);
%     plot(wp(i,:),'-*');
% end

%%%%%%%%%%%%%%%%%%Fig5.17%%%%%%%%%%%%%%%%%%%%
nmr=csvread('D:\Program\python\esl\chapter5\nmr1.csv',1,1);
nmr=nmr';
[xden,denoisedcfs,origcfs] = wdenoise(nmr,6,'Wavelet','sym8','DenoisingMethod','SURE');
figure;
subplot(8,2,1);
plot(nmr);
title('Original');
for i=7:-1:1
    subplot(8,2,2*(8+1-i)-1);
    stem(origcfs{i},'Marker','none');
end
subplot(8,2,2);
plot(xden);
title('WaveShrunk');
for i=7:-1:1
    subplot(8,2,2*(8+1-i));
    stem(denoisedcfs{i},'Marker','none');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%Fig5.19 up%%%%%%%%%%%%%%
pp = csaps(1:length(nmr),nmr,0.5);%smoothing spline
points=fnplt(pp);
figure;
plot(nmr,'k');hold on
plot(points(1,:),points(2,:),'b','LineWidth',2);hold on
plot(xden,'r','LineWidth',2);
legend('Original','Smoothing Spline','Wavelets-sym8')
axis tight
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%Fig 5.19 bottom%%%%%%%%%%%%%%%
x=linspace(0,1,1024);
y=sin(12*(x+0.2))./ (x+0.2);
observedy=y+randn(1,length(y));
[xden,denoisedcfs,origcfs] = wdenoise(observedy,6,'Wavelet','sym8','DenoisingMethod','SURE');
pp = csaps(x,observedy,0.99);%R语言会自动使用交叉验证计算惩罚系数，这里是随意指定的
points=fnval(pp,x);
figure;
plot(x,y,'k','LineWidth',2);hold on
plot(x,observedy,'k*');hold on
plot(x,points,'b','LineWidth',2);hold on
plot(x,xden,'r','LineWidth',2);
legend('Original','Observed','Smoothing Spline','Wavelets-sym8')
xlabel('x1');
ylabel('x2');
axis tight