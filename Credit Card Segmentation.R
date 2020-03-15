#Clear the working Environment
rm(list=ls())

#Set the working path
setwd("D:/Data Science Edwisor/Project")

#Get the working Directory
getwd()

# Importing File
credit=read.csv("credit-card-data.csv",stringsAsFactors = FALSE)

View(credit)

# Identifying Outliers
mystats = function(x) { #declairing the functions
  nmiss=sum(is.na(x))#count the number of na in x and sum it
  a = x[!is.na(x)]# return all the non- NA elements of x
  m = mean(a) # finding the mean
  n = length(a) # finding the length of a
  s = sd(a) #standard deviation of a
  min = min(a) # finding the minimum value a
  p1=quantile(a,0.01) # quantile: divided into equal-sized
  p5=quantile(a,0.05) # quantile: divided into equal-sized
  p10=quantile(a,0.10) # quantile: divided into equal-sized
  q1=quantile(a,0.25) # quantile: divided into equal-sized
  q2=quantile(a,0.5) # quantile: divided into equal-sized
  q3=quantile(a,0.75) # quantile: divided into equal-sized
  p90=quantile(a,0.90) # quantile: divided into equal-sized
  p95=quantile(a,0.95) # quantile: divided into equal-sized
  p99=quantile(a,0.99) # quantile: divided into equal-sized
  max = max(a) # finding the maximum value of a
  UC = m+2*s 
  LC = m-2*s
  outlier_flag= max>UC | min<LC
  return(c(n=n, nmiss=nmiss, outlier_flag=outlier_flag, mean=m, stdev=s,min = min, p1=p1,p5=p5,p10=p10,q1=q1,q2=q2,q3=q3,p90=p90,p95=p95,p99=p99,max=max, UC=UC, LC=LC ))
}
#New Variables creation

#Calculating Monthly_Avg_PURCHASES
credit$Monthly_Avg_PURCHASES <- credit$PURCHASES/(credit$PURCHASES_FREQUENCY*credit$TENURE)

#Calculating Monthly cash advance
credit$Monthly_CASH_ADVANCE <- credit$CASH_ADVANCE/(credit$CASH_ADVANCE_FREQUENCY*credit$TENURE)

#Calculating Limit Usage
credit$LIMIT_USAGE <- credit$BALANCE/credit$CREDIT_LIMIT

# calculating Payment Ratio
credit$MIN_PAYMENTS_RATIO <- credit$PAYMENTS/credit$MINIMUM_PAYMENTS

#Creating csv file with name New_variables_creation.csv
write.csv(credit,"New_variables_creation.csv")

# Combining all the heading and storing in Num_Vars
Num_Vars = c(
  "BALANCE",
  "BALANCE_FREQUENCY",
  "PURCHASES",
  "Monthly_Avg_PURCHASES",
  "ONEOFF_PURCHASES",
  "INSTALLMENTS_PURCHASES",
  "CASH_ADVANCE",
  "Monthly_CASH_ADVANCE",
  "PURCHASES_FREQUENCY",
  "ONEOFF_PURCHASES_FREQUENCY",
  "PURCHASES_INSTALLMENTS_FREQUENCY",
  "CASH_ADVANCE_FREQUENCY",
  "CASH_ADVANCE_TRX",
  "PURCHASES_TRX",
  "CREDIT_LIMIT",
  "LIMIT_USAGE",
  "PAYMENTS",
  "MINIMUM_PAYMENTS",
  "MIN_PAYMENTS_RATIO",
  "PRC_FULL_PAYMENT",
  "TENURE")

# apply() function splits up the matrix in rows
Outliers=t(data.frame(apply(credit[Num_Vars], 2, mystats)))

# viewing the outliers
View(Outliers)

#Creating outliers data in csv file as "Outliers.csv"  
write.csv(Outliers,"Outliers.csv")

# Outlier Treatment
#purifying the outliers 
#Asssigning the UC values for outlier treatment
credit$BALANCE[credit$BALANCE>5727.53]<-5727.53# UC values
credit$BALANCE_FREQUENCY[credit$BALANCE_FREQUENCY>1.3510787]<-1.3510787
credit$PURCHASES[credit$PURCHASES>5276.46]<-5276.46
credit$Monthly_Avg_PURCHASES[credit$Monthly_Avg_PURCHASES>800.03] <- 800.03
credit$ONEOFF_PURCHASES[credit$ONEOFF_PURCHASES>3912.2173709]<-3912.2173709
credit$INSTALLMENTS_PURCHASES[credit$INSTALLMENTS_PURCHASES>2219.7438751]<-2219.7438751
credit$CASH_ADVANCE[credit$CASH_ADVANCE>5173.1911125]<-5173.1911125
credit$Monthly_CASH_ADVANCE[credit$Monthly_CASH_ADVANCE>2558.53] <- 2558.53
credit$PURCHASES_FREQUENCY[credit$PURCHASES_FREQUENCY>1.2930919]<-1.2930919
credit$ONEOFF_PURCHASES_FREQUENCY[credit$ONEOFF_PURCHASES_FREQUENCY>0.7991299]<-0.7991299
credit$PURCHASES_INSTALLMENTS_FREQUENCY[credit$PURCHASES_INSTALLMENTS_FREQUENCY>1.1593329]<-1.1593329
credit$CASH_ADVANCE_FREQUENCY[credit$CASH_ADVANCE_FREQUENCY>0.535387]<-0.535387
credit$CASH_ADVANCE_TRX[credit$CASH_ADVANCE_TRX>16.8981202]<-16.8981202
credit$PURCHASES_TRX[credit$PURCHASES_TRX>64.4251306]<-64.4251306
credit$CREDIT_LIMIT[credit$CREDIT_LIMIT>11772.09]<-11772.09
credit$LIMIT_USAGE[credit$LIMIT_USAGE>1.1683] <- 1.1683
credit$PAYMENTS[credit$PAYMENTS>7523.26]<-7523.26
credit$MINIMUM_PAYMENTS[credit$MINIMUM_PAYMENTS>5609.1065423]<-5609.1065423
credit$MIN_PAYMENTS_RATIO[credit$MIN_PAYMENTS_RATIO>249.9239] <- 249.9239
credit$PRC_FULL_PAYMENT[credit$PRC_FULL_PAYMENT>0.738713]<-0.738713
credit$TENURE[credit$TENURE>14.19398]<-14.19398

# Missing Value Imputation with mean # with mean
credit$MINIMUM_PAYMENTS[which(is.na(credit$MINIMUM_PAYMENTS))] <- 721.9256368
credit$CREDIT_LIMIT[which(is.na(credit$CREDIT_LIMIT))] <- 4343.62
credit$Monthly_Avg_PURCHASES[which(is.na(credit$Monthly_Avg_PURCHASES))] <-184.8991609
credit$Monthly_CASH_ADVANCE[which(is.na(credit$Monthly_CASH_ADVANCE))] <- 717.7235629
credit$LIMIT_USAGE[which(is.na(credit$LIMIT_USAGE))] <-0.3889264
credit$MIN_PAYMENTS_RATIO[which(is.na(credit$MIN_PAYMENTS_RATIO))]  <- 9.3500701

# Checking Missing Value
check_Missing_Values<-t(data.frame(apply(credit[Num_Vars], 2, mystats)))

View(check_Missing_Values)

write.csv(credit,"Missing_value_treatment.csv")

# Variable Reduction (Factor Analysis)
Step_nums <- credit[Num_Vars]
# computes the correlation coefficient of Step_nums
corrm<- cor(Step_nums)

View(corrm)

write.csv(corrm, "Correlation_matrix.csv")

#Installing the required packages installr for updating the r version
install.packages("installr");
library(installr)
#Udating the r
updateR()


#Eigen values for stability analysis 
eigen(corrm)$values

#In order to use the mutate function, we need to install the dplyr package
require(dplyr)

# the mutate function is used to create a new variable from a data set.
eigen_values <- mutate(data.frame(eigen(corrm)$values)
                       ,cum_sum_eigen=cumsum(eigen.corrm..values)
                       , pct_var=eigen.corrm..values/sum(eigen.corrm..values)
                       , cum_pct_var=cum_sum_eigen/sum(eigen.corrm..values))


write.csv(eigen_values, "EigenValues2.csv")

# doing the kind of basic data analysis and psychometric analysis that psychologists
install.packages("psych")
require(psych)

#fa=factor analysis algorithm
# varimax rotation is used to simplify the expression of a particular sub-space in terms of just a few major items
# maximum likelihood(ml) solution produce more expansive output.

FA=fa(r=corrm, 7, rotate="varimax", fm="ml")  

#SORTING THE LOADINGS
FA_SORT<-fa.sort(FA)
# 'Loadings' is a term from factor analysis
FA_SORT$loadings

# Loading and sorting ml data
Loadings<-data.frame(FA_SORT$loadings[1:ncol(Step_nums),])

# saving Loading data in the name of loadings2.csv file
write.csv(Loadings, "loadings2.csv")


# standardizing the data
segment_prepared <-credit[Num_Vars]

#scale , with default settings, will calculate the mean and standard deviation of the entire vector, 
#then "scale" each element by those values by subtracting the mean and dividing by the sd.
segment_prepared = scale(segment_prepared)

write.csv(segment_prepared, "standardized data.csv")

# building clusters using k-means clustering 
# K-means clustering can be used to classify observations into k groups, based on their similarity
cluster_three <- kmeans(segment_prepared,3)
cluster_four <- kmeans(segment_prepared,4)
cluster_five <- kmeans(segment_prepared,5)
cluster_six <- kmeans(segment_prepared,6)

#$ is used to extract the column  as a vector
credit_new<-cbind(credit,km_clust_3=cluster_three$cluster,km_clust_4=cluster_four$cluster,km_clust_5=cluster_five$cluster ,km_clust_6=cluster_six$cluster   )


View(credit_new)

# Profiling
# Combining all the data  
Num_Vars2 <- c(
  "Monthly_Avg_PURCHASES",
  "Monthly_CASH_ADVANCE",
  "CASH_ADVANCE",
  "CASH_ADVANCE_TRX",
  "CASH_ADVANCE_FREQUENCY",
  "ONEOFF_PURCHASES",
  "ONEOFF_PURCHASES_FREQUENCY",
  "PAYMENTS",
  "CREDIT_LIMIT",
  "LIMIT_USAGE",
  "PURCHASES_INSTALLMENTS_FREQUENCY",
  "PURCHASES_FREQUENCY",
  "INSTALLMENTS_PURCHASES",
  "PURCHASES_TRX",
  "MINIMUM_PAYMENTS",
  "MIN_PAYMENTS_RATIO",
  "BALANCE",
  "TENURE"
)
#install.packages("table", dependencies=TRUE, repos='http://cran.rstudio.com/')

install.packages("table")

# The require() is designed to be used inside functions as it gives a warning message and 
#returns a logical value say, FALSE if the requested package is not found and TRUE if the package is loaded.
require(tables)

tt =cbind(tabular(1+factor(km_clust_3)+factor(km_clust_4)+factor(km_clust_5)+
                     factor(km_clust_6)~Heading()*length*All(credit[1]),
                   data=credit_new),tabular(1+factor(km_clust_3)+factor(km_clust_4)+factor(km_clust_5)+
                                              factor(km_clust_6)~Heading()*mean*All(credit[Num_Vars2]),
                                            data=credit_new))

tt2 <- as.data.frame.matrix(tt)
View(tt2)

# rownames in tt2 dataframe
rownames(tt2)<-c(
  "ALL",
  "KM3_1",
  "KM3_2",
  "KM3_3",
  "KM4_1",
  "KM4_2",
  "KM4_3",
  "KM4_4",
  "KM5_1",
  "KM5_2",
  "KM5_3",
  "KM5_4",
  "KM5_5",
  "KM6_1",
  "KM6_2",
  "KM6_3",
  "KM6_4",
  "KM6_5",
  "KM6_6")

# colnames in tt2 dataframe
colnames(tt2)<-c(
  "SEGMENT_SIZE",
  "Monthly_Avg_PURCHASES",
  "Monthly_CASH_ADVANCE",
  "CASH_ADVANCE",
  "CASH_ADVANCE_TRX",
  "CASH_ADVANCE_FREQUENCY",
  "ONEOFF_PURCHASES",
  "ONEOFF_PURCHASES_FREQUENCY",
  "PAYMENTS",
  "CREDIT_LIMIT",
  "LIMIT_USAGE",
  "PURCHASES_INSTALLMENTS_FREQUENCY",
  "PURCHASES_FREQUENCY",
  "INSTALLMENTS_PURCHASES",
  "PURCHASES_TRX",
  "MINIMUM_PAYMENTS",
  "MIN_PAYMENTS_RATIO",
  "BALANCE",
  "TENURE"
)

# Matrix Transpose. ... frame x , t returns the transpose of x .
cluster_profiling2 <- t(tt2)

write.csv(cluster_profiling2,'cluster_profiling2.csv')

