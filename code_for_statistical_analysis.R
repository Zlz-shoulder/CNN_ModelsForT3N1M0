#trainData_EBV: training cohort
#valData_EBV: test cohort

#buliding Model_clinc using the multivariable CPH method
model2 <- coxph(Surv(DFS.time,DFS.event) ~smoking + drinking + sex + age + EBV_4k + LDH + CRP
                , data = trainData_EBV)
summary(model2)
#backward stepwise selection
cox2<-step(model2,direction = c("backward"))#with a default of "both"
summary(cox2)
tra1 <- predict(cox2, newdata = trainData_EBV,type = "lp")
cd = concordance.index(x = (tra1), surv.time=trainData_EBV$DFS.time, surv.event=trainData_EBV$DFS.event,method = "noether")
cd[1:6]
trainData_EBV$clin_sig = tra1

val_pred = predict(cox2, newdata = valData_EBV,type = "lp")
cd = concordance.index(x = (val_pred), surv.time = valData_EBV$DFS.time, surv.event = valData_EBV$DFS.event,method = "noether")
cd[1:6]
valData_EBV$clin_sig = val_pred

#building Model_clinic+dl
model3 <- coxph(Surv(DFS.time,DFS.event) ~smoking + drinking + sex + age + LDH + CRP+ EBV_4k + DL_pred_T1 + DL_pred_T2 + DL_pred_T1C, data = trainData_EBV)
summary(model3)
model3<-step(model3,direction = c("backward"))#with a default of "both"
summary(model3)
tra1 <- predict(model3, newdata = trainData_EBV,type = "lp")
cd = concordance.index(x = (tra1), surv.time=trainData_EBV$DFS.time, surv.event=trainData_EBV$DFS.event,method = "noether")
cd[1:6]

val_pred = predict(model3, newdata = valData_EBV,type = "lp")
cd = concordance.index(x = (val_pred), surv.time = valData_EBV$DFS.time, surv.event = valData_EBV$DFS.event,method = "noether")
cd[1:6]

trainData_EBV$nomo_sig = tra1
valData_EBV$nomo_sig = val_pred

#finding the optimal cutoff
library(survMisc)
z1 = trainData_EBV$nomo_sig
b2 = data.frame(z1)
b2$t2 = trainData_EBV$DFS.time
b2$d3 = trainData_EBV$DFS.event
coxm1 <- coxph(Surv(t2, d3) ~ z1, data = b2)
coxm1 <- cutp(coxm1)$z1
cutoff = round(coxm1$z1[1],3)

trainData_EBV$stra = ifelse(trainData_EBV$nomo_sig < cutoff,0,1)
valData_EBV$stra = ifelse(valData_EBV$nomo_sig < cutoff,0,1)

#risk stratification using Model_clinic+dl in the training cohort
#for DFS
dd<-data.frame("surv.time" = trainData_EBV$DFS.time, "surv.event" = trainData_EBV$DFS.event,"strat" = trainData_EBV$stra)

km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

#for OS
dd<-data.frame("surv.time" = trainData_EBV$OS.time, "surv.event" = trainData_EBV$OS.event,"strat" = trainData_EBV$stra)

km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

#for DMFS
dd<-data.frame("surv.time" = trainData_EBV$DMFS.time, "surv.event" = trainData_EBV$DMFS.event,"strat" = trainData_EBV$stra)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

#for LRFS
dd<-data.frame("surv.time" = trainData_EBV$LRFS.time, "surv.event" = trainData_EBV$LRFS.event,"strat" = trainData_EBV$stra)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))


#risk stratification using Model_clinic+dl in the test cohort
#for DFS
dd<-data.frame("surv.time" = valData_EBV$DFS.time, "surv.event" = valData_EBV$DFS.event,"strat" = valData_EBV$stra)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))
                                                                                         
#for OS                                                                                      
dd<-data.frame("surv.time" = valData_EBV$OS.time, "surv.event" = valData_EBV$OS.event,"strat" = valData_EBV$stra)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

#for DMFS
dd<-data.frame("surv.time" = valData_EBV$DMFS.time, "surv.event" = valData_EBV$DMFS.event,"strat" = valData_EBV$stra)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

#for LRFS
dd<-data.frame("surv.time" = valData_EBV$LRFS.time, "surv.event" = valData_EBV$LRFS.event,"strat" = valData_EBV$stra)
km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
              x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
              leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
              , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

#comparing two different c-indices
library(survcomp)
tra1 <- trainData_EBV$nomo_sig
cd1 = concordance.index(x = (tra1), surv.time=trainData_EBV$DFS.time, surv.event=trainData_EBV$DFS.event,method = "noether")
cd1$c.index
tra2 <- trainData_EBV$clin_sig
cd2 = concordance.index(x = (tra2), surv.time=trainData_EBV$DFS.time, surv.event=trainData_EBV$DFS.event,method = "noether")
cd2$c.index
cindex.comp(cd1, cd2)

#Plotting TD-ROC curvs
#training cohort
nob1<-NROW(trainData_EBV$rad_sig)
t.1<-survivalROC(Stime = trainData_EBV$DFS.time,status = trainData_EBV$DFS.event,(trainData_EBV$clin_sig),predict.time = 1080, span=0.001*nob1^(-0.2))
x1 = round(t.1$AUC,3)
x1 = paste("AUC =",as.character(x1), sep = " ")
t.2<-survivalROC(Stime = trainData_EBV$DFS.time,status = trainData_EBV$DFS.event,(trainData_EBV$DL_pred_T1),predict.time = 1080, span=0.001*nob1^(-0.2))
x2 = round(t.2$AUC,3)
x2 = paste("AUC =",as.character(x2), sep = " ")
t.3<-survivalROC(Stime = trainData_EBV$DFS.time,status = trainData_EBV$DFS.event,(trainData_EBV$DL_pred_T2),predict.time = 1080, span=0.001*nob1^(-0.2))
x3 = round(t.3$AUC,3)
x3 = paste("AUC =",as.character(x3), sep = " ")
t.4<-survivalROC(Stime = trainData_EBV$DFS.time,status = trainData_EBV$DFS.event,(trainData_EBV$DL_pred_T1C),predict.time = 1080, span=0.001*nob1^(-0.2))
x4 = round(t.4$AUC,3)
x4 = paste("AUC =",as.character(x4), sep = " ")
t.5<-survivalROC(Stime = trainData_EBV$DFS.time,status = trainData_EBV$DFS.event,(trainData_EBV$nomo_sig),predict.time = 1080, span=0.001*nob1^(-0.2))
x5 = round(t.5$AUC,3)
x5 = paste("AUC =",as.character(x5), sep = " ")
dev.new()
plot(t.1$FP, t.1$TP, type="l", xlim=c(0,1), ylim=c(0,1),xlab = c("False positive rate (%)"),ylab="True positive rate (%)"
     , lwd = 2, cex.lab=1.5, col = "#000000")
legend("bottomright", legend=c(x1,x2,x3,x4,x5), col =  c("#000000","#56B4E9", "#009E73", "#F0E442", "red"),lwd=2, cex=1.5)
lines(c(0,1), c(0,1), lty = 6,col = rgb(113/255,150/255,159/255),lwd=2.0)#画45度基
lines(t.2$FP,t.2$TP,lty = 1,lwd =2, col = "#56B4E9")
lines(t.3$FP,t.3$TP,lty = 1,lwd =2, col = "#009E73") 
lines(t.4$FP,t.4$TP,lty = 1,lwd =2, col = "#F0E442")
lines(t.5$FP,t.5$TP,lty = 1,lwd =2, col = "red") 

#test cohort
nob1<-NROW(valData_EBV$rad_sig)
t.1<-survivalROC(Stime = valData_EBV$DFS.time,status = valData_EBV$DFS.event,(valData_EBV$clin_sig),predict.time = 1080, span=0.001*nob1^(-0.2))
x1 = round(t.1$AUC,3)
x1 = paste("AUC =",as.character(x1), sep = " ")
t.2<-survivalROC(Stime = valData_EBV$DFS.time,status = valData_EBV$DFS.event,(valData_EBV$DL_pred_T1),predict.time = 1080, span=0.001*nob1^(-0.2))
x2 = round(t.2$AUC,3)
x2 = paste("AUC =",as.character(x2), sep = " ")
t.3<-survivalROC(Stime = valData_EBV$DFS.time,status = valData_EBV$DFS.event,(valData_EBV$DL_pred_T2),predict.time = 1080, span=0.001*nob1^(-0.2))
x3 = round(t.3$AUC,3)
x3 = paste("AUC =",as.character(x3), sep = " ")
t.4<-survivalROC(Stime = valData_EBV$DFS.time,status = valData_EBV$DFS.event,(valData_EBV$DL_pred_T1C),predict.time = 1080, span=0.001*nob1^(-0.2))
x4 = round(t.4$AUC,3)
x4 = paste("AUC =",as.character(x4), sep = " ")
t.5<-survivalROC(Stime = valData_EBV$DFS.time,status = valData_EBV$DFS.event,(valData_EBV$nomo_sig),predict.time = 1080, span=0.001*nob1^(-0.2))
x5 = round(t.5$AUC,3)
x5 = paste("AUC =",as.character(x5), sep = " ")
dev.new()
plot(t.1$FP, t.1$TP, type="l", xlim=c(0,1), ylim=c(0,1),xlab = c("False positive rate (%)"),ylab="True positive rate (%)"
     , lwd = 2, cex.lab=1.5, col = "#000000")
legend("bottomright", legend=c(x1,x2,x3,x4,x5), col =  c("#000000","#56B4E9", "#009E73", "#F0E442", "red"),lwd=2, cex=1.5)
lines(c(0,1), c(0,1), lty = 6,col = rgb(113/255,150/255,159/255),lwd=2.0)#画45度基
lines(t.2$FP,t.2$TP,lty = 1,lwd =2, col = "#56B4E9")
lines(t.3$FP,t.3$TP,lty = 1,lwd =2, col = "#009E73") 
lines(t.4$FP,t.4$TP,lty = 1,lwd =2, col = "#F0E442")
lines(t.5$FP,t.5$TP,lty = 1,lwd =2, col = "red")

#visualizing Model_clinic+dl into the radiomic nonogram
library(lattice);library(survival);library(Formula);library(ggplot2);library(Hmisc);library(rms)

ddist0 <- datadist(trainData_EBV)
options(datadist='ddist0')
f <- cph(Surv(DFS.time, DFS.event) ~ age + LDH + EBV_4k + DL_pred_T1 + DL_pred_T2 + DL_pred_T1C, surv = TRUE, x = T, y = T, data = trainData_EBV)
surv.prob <- Survival(f) # 构建生存概率函数
nom <- nomogram(f, fun=list(function(x) surv.prob(1080, x),function(x) surv.prob(1800, x)),
                funlabel=c("3-year DFS rate","5-year DFS rate"),
                fun.at=c(0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0),
                lp=F)
dev.new()
plot(nom, xfrac=.3,cex.axis=1.3, cex.var=1.3)

#plotting calibration curves of Model_clinic+dl
## training cohort
f_3 <- cph(Surv(DFS.time, DFS.event) ~ nomo_sig, surv = TRUE, x = T, y = T, data = trainData_EBV
           ,time.inc = 1080)
cal_3 <- calibrate(f_3,  u=1080, cmethod='KM',method='boot', B = 50,m=140,surv=TRUE, time.inc=1080) 
f_5 <- cph(Surv(DFS.time, DFS.event) ~ nomo_sig, surv = TRUE, x = T, y = T, data = trainData_EBV
           ,time.inc = 1800,identity.lty=2)
cal_5 <- calibrate(f_5,  u=1800, cmethod='KM',method='boot', B = 50,m=140,surv=TRUE, time.inc=1800) 
source("HL-test.r") #Hosmer-Lemeshow test
y1 = HLtest(cal_3)
y1 = paste("3-year DFS: p =",as.character(round(y1,2)), sep = " ")
y2 = HLtest(cal_5)
y2 = paste("5-year DFS: p =",as.character(round(y2,2)), sep = " ")
dev.new()
opar <- par(no.readonly = TRUE)
par(lwd = 1.2, lty = 1)
x1 = c(rgb(0,112,255,maxColorValue = 255))
xm = 0.5
plot(cal_3,lty = 1,pch = 16,errbar.col = x1,par.corrected = list(col=x1),conf.int=T,lwd = 1.2
     ,xlim = c(xm,1),ylim = c(xm,1),riskdist = F,col = "blue",axes = F)
x2 = c(rgb(220,220,220,maxColorValue = 255))
abline(0, 1, lty = 5, col=x2 ,lwd=1)
x2 = c(rgb(209,73,85,maxColorValue = 255))
plot(cal_5,lty = 1,pch = 18,errbar.col = x2,xlim = c(xm,1),ylim = c(xm,1),par.corrected = list(col=x2),
     col = "red",lwd = 1.2,riskdist = F,add=T,conf.int=T)
axis(1,at=seq(xm,1.0,0.1),labels=seq(xm,1.0,0.1),pos=xm)
axis(2,at=seq(xm,1.0,0.1),labels=seq(xm,1.0,0.1),pos=xm)
legend("top", legend=c(y1,y2),col=c('blue','red'), lwd=2, cex=1.5,lty=c(1,1))
par(opar)

## test cohort
f_3 <- cph(Surv(DFS.time, DFS.event) ~ nomo_sig, surv = TRUE, x = T, y = T, data = valData_EBV
           ,time.inc = 1080)
cal_3 <- calibrate(f_3,  u=1080, cmethod='KM',method='boot', B = 50,m = 62,surv=TRUE, time.inc=1080) 
f_5 <- cph(Surv(DFS.time, DFS.event) ~ nomo_sig, surv = TRUE, x = T, y = T, data = valData_EBV
           ,time.inc = 1800)
cal_5 <- calibrate(f_5,  u=1800, cmethod='KM',method='boot', B = 50,m = 62,surv=TRUE, time.inc=1800) 
source("HL-test.r") #Hosmer-Lemeshow test
y1 = HLtest(cal_3)
y1 = paste("3-year DFS: p =",as.character(round(y1,2)), sep = " ")
y2 = HLtest(cal_5)
y2 = paste("5-year DFS: p =",as.character(round(y2,2)), sep = " ")
# dev.new()
opar <- par(no.readonly = TRUE)
par(lwd = 1.2, lty = 1)
x1 = c(rgb(0,112,255,maxColorValue = 255))
xm = 0.5
plot(cal_3,lty = 1,pch = 16,errbar.col = x1,par.corrected = list(col=x1),conf.int=T,lwd = 1.2
     ,xlim = c(xm,1),ylim = c(xm,1),riskdist = F,col = "blue",axes = F)
x2 = c(rgb(220,220,220,maxColorValue = 255))
abline(0, 1, lty = 5, col=x2 ,lwd=1)
x2 = c(rgb(209,73,85,maxColorValue = 255))
plot(cal_5,lty = 1,pch = 18,errbar.col = x2,xlim = c(xm,1),ylim = c(xm,1),par.corrected = list(col=x2),
     col = "red",lwd = 1.2,riskdist = F,add=T,conf.int=T)
axis(1,at=seq(xm,1.0,0.1),labels=seq(xm,1.0,0.1),pos=xm)
axis(2,at=seq(xm,1.0,0.1),labels=seq(xm,1.0,0.1),pos=xm)
legend("top", legend=c(y1,y2),col=c('blue','red'), lwd=2, cex=1.5,lty=c(1,1))
par(opar)
