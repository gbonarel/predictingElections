library(corrr)
library(ggcorrplot)
library(factoextra)

#import Data
data_2020 <- as.data.frame(read.csv("IE490Dataset(1)(2020 Data).csv"))

#check for NA
colSums(is.na(data_2020))
data_2020 <- na.omit(data_2020)
colSums(is.na(data_2020))

#Only Numerical 
normalized_data <- scale(data_2020[3:48])

#Correlation Matrix
corr_matrix <- cor(normalized_data)
ggcorrplot(corr_matrix)

#PRCOMP 
data.pca <- princomp(corr_matrix)
summary(data.pca)

#Scree Plot
png(file = "barplot.png")
fviz_eig(data.pca, addlabels = TRUE)
dev.off()

fviz_pca_var(data.pca, col.var = "black")

fviz_cos2(data.pca, choice = "var", axes = 1:2)

png(file = "combined.png")
fviz_pca_var(data.pca, col.var = "cos2",
             gradient.cols = c("black", "orange", "green"),
             repel = TRUE)
dev.off()