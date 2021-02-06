# This code plots drum/ta classifier, and runs a few statistical tests on some group means.

rm(list=ls())
graphics.off()

#library(Hmisc)
library(ggplot2)

setwd('/home/sam/Dropbox/babyCNN/final_stuff/classification_120epochs')
data_rej_120 <- read.csv('rej1.csv')
data_all_120 <- read.csv('all1.csv')

data_rej_120$auc <- data_rej_120$auc * 100 
data_rej_120$sd <- data_rej_120$sd * 100
data_all_120$auc <- data_all_120$auc * 100
data_all_120$sd <- data_all_120$sd * 100

# plots
plot1 <- ggplot(data_all_120, aes(x=id, y=auc)) +
  geom_errorbar(aes(ymin=auc-sd, ymax=auc+sd), width=1) +
  geom_point() +
  geom_smooth() +  
  theme(axis.text.x=element_blank())+
  labs(y = 'ROC-AUC as %', x = 'Participants') +
  ggtitle('ROBUSTNESS OF THE MODEL TO NOISE') +
  theme(plot.title = element_text(size=25, face='bold', hjust=0.5)) +
  ylim(35,103) +
  theme(text = element_text(size=20)) +
  geom_hline(yintercept=87.4, linetype="dashed", color = "red", size=2) +
  geom_hline(yintercept=58, linetype="dashed", color = "blue", size=1) +
  geom_hline(yintercept=68, linetype="dashed", color = "green", size=1)

plot2 <- ggplot(data_rej_120, aes(x=id, y=auc)) +
  geom_errorbar(aes(ymin=auc-sd, ymax=auc+sd), width=1) +
  geom_point() +
  theme(axis.text.x=element_blank())+
  labs(y = 'ROC-AUC as %', x = 'Participants') +
  ggtitle('CNN RESULTS') +
  theme(plot.title = element_text(size=25, face='bold', hjust=0.5)) +
  ylim(35,103) +
  theme(text = element_text(size=20)) +
  geom_hline(yintercept=87.5, linetype="dashed", color = "red", size=2) +
  geom_hline(yintercept=58, linetype="dashed", color = "blue", size=1, show.legend = TRUE) +
  geom_hline(yintercept=68, linetype="dashed", color = "green", size=1, show.legend = TRUE)

require(gridExtra)
plots <- grid.arrange(plot2, plot1, ncol=2)

# means and sd
mean(data_rej_120$auc)
sd(data_rej_120$auc)

mean(data_all_120$auc)
sd(data_all_120$auc)

# test for normality, data not normal so use non-parametric tests
hist(data_all_120$acc)
hist(data_rej_120$acc)
shapiro.test(data_all_120$acc)
shapiro.test(data_rej_120$acc)

# wilcox tests comparing group means
test <- wilcox.test(data_all_120$acc, data_rej_120$acc, paired=TRUE)
test
Zstat <- qnorm(test$p.value/2)
Zstat
abs(Zstat)/sqrt(95)

# stats for comparing recording sites and optimise batch
data <- data_rej_120
shapiro.test(data$auc[data$rest==1])
test <- wilcox.test(data$auc[data$rest==1], data$auc[data$optimise==1], paired=FALSE)
test
Zstat <- qnorm(test$p.value/2)
Zstat
abs(Zstat)/sqrt(95)

#install.packages('vioplot')
library(vioplot)

data = read.csv('/home/sam/Dropbox/babyCNN/brain_and_language/vioplot_data.csv')

with(data , vioplot( 
  auc[group=="the_rest"] , auc[group=="HS"], auc[group=="optimise"],  
  col=rgb(0.1,0.4,0.7,0.7) , names=c("Site A recordings \n n=79", "Site B recordings \n n=8","Model development set \n n=8"),
  ylab = 'ROC-AUC'
))


