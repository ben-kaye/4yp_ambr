% uiopen('C:\Users\benka\OneDrive - Nexus365\4YP\Unit tests\Exp-03-10\Part-A-Processed\base.csv',1)


expr = readtable('C:\Users\benka\OneDrive - Nexus365\4YP\Unit tests\Exp-03-14\Pump-through-20\exp_r.csv');
expg = readtable('C:\Users\benka\OneDrive - Nexus365\4YP\Unit tests\Exp-03-14\Pump-through-20\exp_g.csv');
expb = readtable('C:\Users\benka\OneDrive - Nexus365\4YP\Unit tests\Exp-03-14\Pump-through-20\exp_b.csv');             
wells = [ 3 5 6 7 ];
wells = 1:12;
density_raw = [ expr{:,wells}, expg{:,wells}, expb{:,wells}] ;




time = linspace(0,length(density_raw)/180, length(density_raw)); % hrs


% figure(1)
% for k = 1:N
%     color = base{:,12+k};
%     color = string(color);
%     for a=1:numel(color)
% 
%         plot([ time(a) time(a) ] ,[k-1,k],'Color',color(a)) 
% 
%          hold on 
%     end
% end
% hold off

dens1 = density_raw - density_raw(1,:);



dens = dens1 / 255;

N=1;
NL = numel(time);

K = 1;

MA = movmean(dens,K);
diff = dens(2:end,:) - dens(1:end-1,:);

s_dens = [ mean(MA(:,1:length(wells)),2), mean(MA(:,length(wells)+1:2*length(wells)),2), mean(MA(:,2*length(wells)+1:end),2)];

figure(3)
plot(1:254,s_dens)
colororder({'r','g','b'})
legend('R avg','G avg', 'B avg')
yline(0,'r--')

ylims = [ min(min(dens)), max(max(dens))];

figure(2)
subplot(3,1,1)
plot(time, MA(:,1:length(wells)))
xlabel('time (hrs)')
ylabel('density')
yline(0,'r--')
legend()
xlim([0, max(time)])
ylim(ylims)
title('R density')
subplot(3,1,2)
plot(time, MA(:,length(wells)+1:2*length(wells)))
xlabel('time (hrs)')
yline(0,'r--')
ylabel('density')
xlim([0, max(time)])
ylim(ylims) 
title('G density')
subplot(3,1,3)
plot(time, MA(:,2*length(wells)+1:end))
xlabel('time (hrs)')
yline(0,'r--')
ylabel('density')
xlim([0, max(time)])
ylim(ylims)
title('B density')