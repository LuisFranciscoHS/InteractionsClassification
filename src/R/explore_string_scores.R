
dat <- read.table("COG.links.detailed.v11.0.txt")
colnames(dat)

head(dat)
summary(as.numeric(dat$V10[-1]))
plot(density(as.numeric(dat$V10[-1])), ylim = c(0, 0.05))