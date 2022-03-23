loc = 'D:\Ben-Kaye-4YP-Temp\Exp-22-03-ANALYSIS\';

expr = readtable(append(loc,'exp_r.csv'));
expg = readtable(append(loc,'exp_g.csv'));
expb = readtable(append(loc,'exp_b.csv'));  

wells = [ 3 5 6 7 ];
wells = 2:13;

density_raw = [ expr{:,wells}, expg{:,wells}, expb{:,wells}] ;
time_unordered = expr{:,1};
[ time, reI] = sort(time_unordered);

time = (time - time(1))/60;
density_raw = density_raw(reI,:);
dens = density_raw - density_raw(1,:);

INDEX_OF_INTEREST = 3:427;
INDEX_OF_INTEREST = 1108:size(dens,1);

dens = dens(INDEX_OF_INTEREST,:) - dens(INDEX_OF_INTEREST(1),:);
time = time(INDEX_OF_INTEREST) - time(INDEX_OF_INTEREST(1));


sample_rate = 1; % per min


% time = linspace(0,length(density_raw)/60/sample_rate, length(density_raw)); % hrs


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

N=1;
NL = numel(time);

K = 10;

MA = movmean(dens,K);
diff = dens(2:end,:) - dens(1:end-1,:);

s_dens = [ mean(MA(:,1:length(wells)),2), mean(MA(:,length(wells)+1:2*length(wells)),2), mean(MA(:,2*length(wells)+1:end),2)];

figure(3)
plot(time,s_dens)
colororder({'r','g','b'})
legend('R avg','G avg', 'B avg')
yline(0,'r--')
xlabel('time (hrs)')
ylabel('well mean intensity (px)')
ylims = [ min(min(dens)), max(max(dens))];

figure(2)
subplot(3,1,1)
plot(time, MA(:,1:length(wells)))
xlabel('time (hrs)')
ylabel('density')
yline(0,'r--')
xline(7,'k-.')
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
