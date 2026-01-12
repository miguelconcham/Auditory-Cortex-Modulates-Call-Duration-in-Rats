load table_r2_withds4
figure
n =1;
type_names = {'Offset Suppressed','Offset ACtivated','Pre-Call','Pre-Call', 'Onset Suppressed','Onset Activated', 'Non Responsive','Non Responsive'}
all_r = [];
for sign_s = [-1 1]
    for type_n = 1:4
        subplot(4,2,2*(type_n-1) + n)
        index = table_r2_withds4.Sign ==sign_s & table_r2_withds4.Type == type_n;
       lm_fit = fitlm(table_r2_withds4(index, {'rate','CallLength'}));
       lme = fitlme(table_r2_withds4(index,:), 'rate ~ CallLength + (1|Ds)');
       [R2_marginal, R2_conditional] = lmeR2(lme);
       all_r(2*(type_n-1) + n) = R2_conditional;
       [c,p] = corr(table_r2_withds4.rate(index), table_r2_withds4.CallLength(index));
       plot(lm_fit, 'Marker','.', 'Color', 'k')
       title([type_names{2*(type_n-1) + n}, ' ', num2str(sign_s)  ' r=', num2str(R2_conditional), ' p =', num2str(lme.coefTest)])
       legend('off')
    end
    n = n+1;
end

 figure
 order2plot = [7 3 2 1 6 5];
bar(all_r(order2plot))
xticklabels(type_names(order2plot))

function [R2_marginal, R2_conditional] = lmeR2(lme)
    % Variance components
    varFixed = var(predict(lme,'Conditional',false));  % Fixed effects variance
    varRandom = sum(cell2mat(lme.covarianceParameters));  % Random effects variance
    varResidual = lme.MSE;  % Residual variance
    
    % Marginal and conditional R^2
    R2_marginal = varFixed / (varFixed + varRandom + varResidual);
    R2_conditional = (varFixed + varRandom) / (varFixed + varRandom + varResidual);
end