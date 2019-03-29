rm(list=ls())
graphics.off()
library(Hmisc)
library(ggplot2)

getwd()
setwd('C:/Users/cnebabylab/Downloads')

redcap <- read.csv('BabyRhythm-EegDataSam_DATA_2019-03-26_0938.csv')
redcap <- redcap[-c(1,3,5,16,19,22,31,34,37,38,39,40,42,49,50,53,54,59,60,62,64,68,79,81,87,91,105,107,108),]
rownames(redcap) <- seq(length=nrow(redcap))
subject <- read.csv('subject.csv', header=F)
redcap$subject <- subject$V1

results <- read.csv('results_2.csv', header=F)
redcap$accuracy <- results[,1]
redcap$SD <- results[,2]

cdi=read.csv('BabyRhythm-EegDataSam_DATA_2019-03-27_1626.csv', header=T)

merged <- merge(redcap, cdi, by='record_id')

merged$diagnosis <- merged$arhq_27 + merged$arhq_27_v2

merged$asqTotal <- merged[,16] + merged[,17] + merged[,18] + merged[,19] + merged[,20]

ggplot(redcap, aes(x=asq2mo_comm, y=accuracy), na.rm=T) + geom_point(shape=1) + geom_smooth(method=lm, color="red", se=TRUE) 

ggplot(merged, aes(x=accuracy, y=cdishort_prod_percentage), na.rm=T) + geom_point(shape=1) + geom_smooth(method=lm, color="red", se=TRUE)

ggplot(merged, aes(x=accuracy, y=cdishort_b_total), na.rm=T) + geom_point(shape=1) + geom_smooth(method=lm, color="red", se=TRUE)

ggplot(merged, aes(x=accuracy, y=gest), na.rm=T) + geom_point(shape=1) + geom_smooth(method=lm, color="red", se=TRUE)

ggplot(merged, aes(x=accuracy, y=cdifull_prod_percentage), na.rm=T) + geom_point(shape=1) + geom_smooth(method=lm, color="red", se=TRUE)

ggplot(merged, aes(x=accuracy, y=diagnosis), na.rm=T) + geom_point(shape=1) + geom_smooth(method=lm, color="red", se=TRUE)

merged <- subset(merged, asqTotal < 500)
ggplot(merged, aes(x=accuracy, y=asqTotal), na.rm=T) + geom_point(shape=1) + geom_smooth(method=lm, color="red", se=TRUE)

lm <- lm(asqTotal ~ accuracy, data=merged)
summary(lm)


f <- ggplot(merged, aes(x=accuracy, y=diagnosis))
f + geom_col()

lm2 <- lm(diagnosis ~ accuracy, data=merged)
summary(lm2)

hist(merged$accuracy, breaks = 10, col='red')
hist(merged$accuracy, breaks=10, col='red')


lm1 <- lm(cdishort_b_total ~ accuracy, data=merged)
summary(lm1)

# read from txt 
results <- read.delim('results.txt', header=F)


plot(merged$accuracy~SD)

ggplot(melt(df, id.vars=c("method", "N")), aes(method, value)) + 
  geom_bar(stat="identity") + facet_wrap(~variable)

ggplot(melt(merged, id.vars='record_id'), aes(record_id, accuracy))

ggplot(merged, aes(x=accuracy, y=SD), na.rm=T) + geom_point(shape=1) + geom_smooth(method=lm, color="red", se=TRUE)

hist(merged$accuracy)
merged$accOrdered <- sort(merged$accuracy)


ggplot(merged, aes(x=record_id, y=accOrdered)) +
  geom_errorbar(aes(ymin=accOrdered-SD, ymax=accOrdered+SD), width=1) +
  geom_line() +
  geom_point() +
  theme(axis.text.x=element_blank()) +
  labs(y = 'Accuracy (%)', x = 'Subjects (N = 94)', title = 'Mean classification accuracy with SD bars (5 fold cross validation)')