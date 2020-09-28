heart<-read.table("D:\\Program\\R\\SAheart.txt",header=TRUE,sep=",")
head(heart)
library(splines)
form = "chd ~ ns(heart$sbp,df=4) + ns(heart$tobacco,df=4) + ns(heart$ldl,df=4) + heart$famhist + ns(heart$obesity,df=4) + ns(heart$age,df=4)"
form = formula(form)
m = glm( form, data=heart, family=binomial )
print( summary(m), digits=3 )
drop1( m, scope=form, test="F" )


SA <- read.table("D:\\Program\\R\\SAheart.txt",header=TRUE,sep=",")
SA.glm <- glm(chd~sbp+tobacco+ldl+famhist+obesity+alcohol+age,family=binomial,data=SA)
summary(SA.glm)
library(splines)
degf  <- 4
X <- cbind(ns(SA$sbp,df=degf),ns(SA$tobacco,df=degf),ns(SA$ldl,df=degf),as.numeric(SA$famhist)-1,ns(SA$obesity,df=degf),ns(SA$age,df=degf))
X <- cbind(rep(1,dim(SA)[1]),scale(X,scale=FALSE))
SA.glm2 <- glm.fit(X,SA$chd,family=binomial())
coeff <- SA.glm2$coefficients

S <-  solve(t(X) %*% diag(SA.glm2$weights) %*% X)
h1 <- X[,2:5] %*% coeff[2:5]
se1 <- sqrt(diag(X[,2:5] %*% S[2:5,2:5] %*% t(X[,2:5])))

h2 <- X[,6:9] %*% coeff[6:9]
se2 <- sqrt(diag(X[,6:9] %*% S[6:9,6:9] %*% t(X[,6:9])))

h3 <- X[,10:13] %*%  coeff[10:13]
se3 <- sqrt(diag(X[,10:13] %*% S[10:13,10:13] %*% t(X[,10:13])))

h5 <- X[,15:18] %*%  coeff[15:18]
se5 <- sqrt(diag(X[,15:18] %*% S[15:18,15:18] %*% t(X[,15:18])))

h6 <- X[,19:22] %*%  coeff[19:22]
se6 <- sqrt(diag(X[,19:22] %*% S[19:22,19:22] %*% t(X[,19:22])))

par(mfrow=c(3,2))
plot(sort(SA$sbp),h1[order(SA$sbp)],type="l",ylim=c(-1,4),xlab="sbp",ylab="")
lines(sort(SA$sbp),h1[order(SA$sbp)]+2*se1[order(SA$sbp)])
lines(sort(SA$sbp),h1[order(SA$sbp)]-2*se1[order(SA$sbp)])
rug(jitter(SA$sbp))

plot(sort(SA$tobacco),h2[order(SA$tobacco)],type="l",ylim=c(-1,8),xlab="tobacco",ylab="")
lines(sort(SA$tobacco),h2[order(SA$tobacco)]+2*se2[order(SA$tobacco)])
lines(sort(SA$tobacco),h2[order(SA$tobacco)]-2*se2[order(SA$tobacco)])
rug(jitter(SA$tobacco))

plot(sort(SA$ldl),h3[order(SA$ldl)],type="l",ylim=c(-4,4),xlab="ldl",ylab="")
lines(sort(SA$ldl),h3[order(SA$ldl)]+2*se3[order(SA$ldl)])
lines(sort(SA$ldl),h3[order(SA$ldl)]-2*se3[order(SA$ldl)])
rug(jitter(SA$ldl))

plot(sort(SA$obesity),h5[order(SA$obesity)],type="l",ylim=c(-2,6),xlab="obesity",ylab="")
lines(sort(SA$obesity),h5[order(SA$obesity)]+2*se5[order(SA$obesity)])
lines(sort(SA$obesity),h5[order(SA$obesity)]-2*se5[order(SA$obesity)])
rug(jitter(SA$obesity))

plot(sort(SA$age),h6[order(SA$age)],type="l",ylim=c(-6,2),xlab="age",ylab="")
lines(sort(SA$age),h6[order(SA$age)]+2*se6[order(SA$age)])
lines(sort(SA$age),h6[order(SA$age)]-2*se6[order(SA$age)])
rug(jitter(SA$age))