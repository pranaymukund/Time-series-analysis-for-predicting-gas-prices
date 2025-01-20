
##### DATA SETS #####
mydata_midgrade <- read.csv("California_Midgrade.csv")

#reordering the rows from bottom to up
mydata_midgrade <-mydata_midgrade %>% arrange(desc(row_number()))

#Chamging the string date to numeric date

# For mydata_midgrade
mydata_midgrade$Month <- as.Date(paste0("1 ", mydata_midgrade$Month), format = "%d %b %Y")
mydata_midgrade$Month <- format(mydata_midgrade$Month, "%m/%d/%Y")
mydata_midgrade



################################################################################
library(forecast)
head(mydata_midgrade)
tail(mydata_midgrade)

# Convert data to a time series object
price_per_galon.ts <- ts(mydata_midgrade$Price.per.galon, start = c(2000, 06), 
                         end = c(2023, 02), freq = 12)
# Create a time series plot
plot(price_per_galon.ts, xlab = "Years", ylab = "Fuel Price Per galon", 
     ylim = c(1, 7), xlim = c(2000, 2023), main= 'MidGrade Fuel Price from 2000 to 2023' )



# Use Acf() function to identify autocorrelation and plot autocorrelation
Acf(price_per_galon.ts, lag.max = 12, main = "Autocorrelation for MidGrade Fuel Price from 2000 to 2023")

# CREATE TIME SERIES PARTITION
nValid <- 60 
nTrain <- length(price_per_galon.ts) - nValid
train.ts <- window(price_per_galon.ts, start = c(2000, 06), end = c(2000, nTrain))
valid.ts <- window(price_per_galon.ts, start = c(2000, nTrain + 1), end = c(2000, nTrain + nValid))

# Use Acf() function to identify autocorrelation for training and validation
Acf(train.ts, lag.max = 12, main = "Autocorrelation for Midgrade Training Data Set")
Acf(valid.ts, lag.max = 12, main = "Autocorrelation for Midgrade Validation Data Set")

################################################################################

# Use tslm() function to create linear trend and seasonal model.
train.lin.season <- tslm(train.ts ~ trend + season)

# See summary of linear trend equation and associated parameters.
summary(train.lin.season)                  


# Apply forecast() function to make predictions for ts with 
# linear trend and seasonal model in validation set.  
train.lin.season.pred <- forecast(train.lin.season, h = nValid, level = 0)
train.lin.season.pred


plot(train.lin.season.pred$mean, 
     xlab = "Months", ylab = "Price per galon", ylim = c(1, 5), 
     bty = "l", xlim = c(2000, 2023), xaxt = "n",
     main = "Regression with Linear Trend and Seasonality", 
     lwd = 2, lty = 2, col = "blue") 

axis(1, at = seq(2000, 2023, 1), labels = format(seq(2000, 2023, 1)))

lines(train.lin.season.pred$fitted, col = "blue", lwd = 2)

lines(train.ts, col = "black", lwd = 2, lty = 1)

lines(valid.ts, col = "black", lwd = 2, lty = 1)

legend(2000,5, legend = c("Price Time Series", "Regression for Training Data",
                          "Forecast for Validation Data"), 
       col = c("black", "blue" , "blue"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")

# Plot on the chart vertical lines and horizontal arrows
# describing training, validation, and future prediction intervals.
lines(c(2018, 2018), c(0, 5))
lines(c(2023, 2023), c(0, 5))
text(2009, 5, "Training")
text(2020.5, 5, "Validation")

# Use Acf() function to identify autocorrelation for the model residuals 
# (training and validation sets), and plot autocorrelation for different 
# lags (up to maximum of 12).
Acf(train.lin.season.pred$residuals, lag.max = 12, 
    main = "Autocorrelation for Mid_Grade Training Residuals")
Acf(valid.ts - train.lin.season.pred$mean, lag.max = 12, 
    main = "Autocorrelation for Mid_Grade Validation Residuals")

################################################################################

# Use Arima() function to fit AR(1) model for training residuals. 
#The Arima model of order = c(1,0,0) gives an AR(1) model.
# Use summary() to identify parameters of AR(1) model. 
res.ar1 <- Arima(train.lin.season$residuals, order = c(1,0,0))
summary(res.ar1)
res.ar1$fitted

# Use forecast() function to make prediction of residuals in validation set.
res.ar1.pred <- forecast(res.ar1, h = nValid, level = 0)
res.ar1.pred

# Develop a data frame to demonstrate the training AR model result svs. 
#original training series, training regression model and its residuals.  
train.df <- round(data.frame(train.ts, train.lin.season$fitted,train.lin.season$residuals, res.ar1$fitted, res.ar1$residuals), 3)
names(train.df) <- c("Price per galon", "Regression", "Residuals","AR.Model", "AR.Model.Residuals")
train.df

# Use Acf() function to identify autocorrelation for the training residual 
#of residuals and plot autocorrelation for different lags(up to maximum of 12).
Acf(res.ar1$residuals, lag.max = 12, main = "Autocorrelation for Mid_Grade Training Residuals of Residuals")

# Create two-level model's forecast with linear trend and seasonality regression + AR(1) for residuals for validation period.
# Create data table with validation data, regression forecast for validation period, AR(1) residuals for validation, and two level model results. 
valid.two.level.pred <- train.lin.season.pred$mean + res.ar1.pred$mean
valid.df <- round(data.frame(valid.ts, train.lin.season.pred$mean,res.ar1.pred$mean, valid.two.level.pred),3)
names(valid.df) <- c("price", "Reg.Forecast","AR(1)Forecast", "Combined.Forecast")
valid.df
################################################################################

# Use accuracy() function to identify common accuracy measures for validation period forecast:
# (1) two-level model (linear trend and seasonal model + AR(1) model for residuals),
# (2) linear trend and seasonality model only.
round(accuracy(valid.two.level.pred, valid.ts), 3)
round(accuracy(train.lin.season.pred$mean, valid.ts), 3)

################################################################################
# FIT REGRESSION MODEL WITH LINEAR TREND AND SEASONALITY FOR ENTIRE DATASET. FORECAST AND PLOT DATA, AND MEASURE ACCURACY.
# Use tslm() function to create linear trend and seasonality model.
lin.season <- tslm(price_per_galon.ts ~ trend + season)
# See summary of linear trend equation and associated parameters.
summary(lin.season)

# Apply forecast() function to make predictions with linear trend and seasonal model into the future 12 months.  
lin.season.pred <- forecast(lin.season, h = 12, level = 0)
lin.season.pred


# Use Acf() function to identify autocorrelation fortS the model residuals 
# (training and validation sets), and plot autocorrelation for different 
# lags (up to maximum of 12).
Acf(train.lin.season.pred$residuals, lag.max = 12, 
    main = "Autocorrelation for Mid_Grade Training Residuals")
Acf(valid.ts - train.lin.season.pred$mean, lag.max = 12, 
    main = "Autocorrelation for Mid_Grade Validation Residuals")


# Use Acf() function to identify autocorrelation for the model residuals for entire data set, and plot autocorrelation for different lags (up to maximum of 12).
Acf(lin.season.pred$residuals, lag.max = 12, main = "Autocorrelation of Regression Residuals for Entire Data Set")

###############################################################################

# Use Arima() function to fit AR(1) model for regression residuals. 
#The ARIMA model order of order = c(1,0,0) gives an AR(1) model. 
#Use forecast() function to make prediction of residuals into the future 12 months.
residual.ar1 <- Arima(lin.season$residuals, order = c(1,0,0))
residual.ar1.pred <- forecast(residual.ar1, h = 12, level = 0)
summary(residual.ar1)

# Use Acf() function to identify autocorrelation for the residuals of residuals 
#and plot autocorrelation for different lags (up to maximum of 12).
Acf(residual.ar1$residuals, lag.max = 12, main = "Autocorrelation for Residuals of Residuals for Entire Data Set")

# Identify forecast for the future 12 periods as sum of linear trend and 
# seasonal model and AR(1) model for residuals.
lin.season.ar1.pred <- (lin.season.pred$mean + residual.ar1.pred$mean)
lin.season.ar1.pred

# Create a data table with linear trend and seasonal forecast for 12 future periods,AR(1) model for residuals for 12 future periods, and combined two-level forecast for 12 future periods. 
table.df <- round(data.frame(lin.season.pred$mean, residual.ar1.pred$mean, lin.season.ar1.pred),3)
names(table.df) <- c("Reg.Forecast", "AR(1)Forecast","Combined.Forecast")
table.df

# Use accuracy() function to identify common accuracy measures for:
# (1) two-level model (linear trend and seasonality model 
#     + AR(1) model for residuals),
# (2) linear trend and seasonality model only, and
# (3) seasonal naive forecast. 
round(accuracy(lin.season$fitted + residual.ar1$fitted, price_per_galon.ts), 3)
round(accuracy(lin.season$fitted, price_per_galon.ts), 3)
round(accuracy((snaive(price_per_galon.ts))$fitted, price_per_galon.ts), 3)


################################################################################################################################################################


################################################################################################################################################################


## USE ts() FUNCTION TO CREATE TIME SERIES DATA SET.
price_per_galon_arima.ts <- ts(mydata_midgrade$Price.per.galon,start = 
                                 c(2000, 06), end = c(2023, 02), freq = 12)

# Use Acf() function to identify autocorrelation and plot autocorrelation
# for different lags (up to maximum of 12).
Acf(price_per_galon_arima.ts, lag.max = 12, main = "Autocorrelation for Price_per_galon")


# CREATE TIME SERIES PARTITION
nValid <- 60 
nTrain <- length(price_per_galon_arima.ts) - nValid
train.ts <- window(price_per_galon_arima.ts, start = c(2000, 06), end = c(2000, nTrain))
valid.ts <- window(price_per_galon_arima.ts, start = c(2000, nTrain + 1), end = c(2000, nTrain + nValid))


################################################################################
## FIT AR(2) MODEL.
# Use Arima() function to fit AR(2) model.
# The ARIMA model of order = c(2,0,0) gives an AR(2) model.
# Use summary() to show AR(2) model and its parameters.
train.ar2 <- Arima(train.ts, order = c(2,0,0))
summary(train.ar2)

# Apply forecast() function to make predictions for ts with 
# AR model in validation set.   
train.ar2.pred <- forecast(train.ar2, h = nValid, level = 0)
train.ar2.pred

# Use Acf() function to create autocorrelation chart of AR(2) model residuals.
Acf(train.ar2$residuals, lag.max = 12, main = 
      "Autocorrelations of AR(2) Model Residuals in Training Period")



################################################################################
## FIT MA(2) MODEL.
# Use Arima() function to fit MA(2) model.
# The ARIMA model of order = c(0,0,2) gives an MA(2) model.
# Use summary() to show MA(2) model and its parameters.
train.ma2<- Arima(train.ts, order = c(0,0,2))
summary(train.ma2)

# Apply forecast() function to make predictions for ts with 
# MA model in validation set.    
train.ma2.pred <- forecast(train.ma2, h = nValid, level = 0)
train.ma2.pred

# Use Acf() function to create autocorrelation chart of MA(2) model residuals.
Acf(train.ma2$residuals, lag.max = 12,  main = "Autocorrelations of MA(2) Model Residuals")


################################################################################
## FIT ARMA(2,2) MODEL.
# Use Arima() function to fit ARMA(2,2) model.
# The ARIMA model of order = c(2,0,2) gives an ARMA(2,2) model.
# Use summary() to show ARMA model and its parameters.
train.arma2 <- Arima(train.ts, order = c(2,0,2), method="ML")
summary(train.arma2)

# Apply forecast() function to make predictions for ts with 
# ARMA model in validation set.    
train.arma2.pred <- forecast(train.arma2, h = nValid, level = 0)
train.arma2.pred

# Use Acf() function to create autocorrelation chart of ARMA(2,2) model residuals.
Acf(train.arma2$residuals, lag.max = 12, main = "Autocorrelations of ARMA(2,2) Model Residuals")


################################################################################
## FIT ARIMA(2,1,2) MODEL.
# Use Arima() function to fit ARIMA(2,1,2) model.
# Use summary() to show ARIMA model and its parameters.
train.arima <- Arima(train.ts, order = c(2,1,2)) 
summary(train.arima)

# Apply forecast() function to make predictions for ts with 
# ARIMA model in validation set.    
train.arima.pred <- forecast(train.arima, h = nValid, level = 0)
train.arima.pred

# Using Acf() function, create autocorrelation chart of ARIMA(2,1,2) model residuals.
Acf(train.arima$residuals, lag.max = 12, main = "Autocorrelations of ARIMA(2,1,2) Model Residuals")


################################################################################
## FIT ARIMA(2,1,2)(1,1,2) MODEL.
# Use Arima() function to fit ARIMA(2,1,2)(1,1,2) model for 
# trend and seasonality.
# Use summary() to show ARIMA model and its parameters.
train.arima.seas <- Arima(train.ts, order = c(2,1,2),seasonal = c(1,1,2)) 
summary(train.arima.seas)


# Apply forecast() function to make predictions for ts with 
# ARIMA model in validation set.    
train.arima.seas.pred <- forecast(train.arima.seas, h = nValid, level = 0)
train.arima.seas.pred

# Use Acf() function to create autocorrelation chart of ARIMA(2,1,2)(1,1,2) 
# model residuals.
Acf(train.arima.seas$residuals, lag.max = 12,main = "Autocorrelations of ARIMA(2,1,2)(1,1,2) Model Residuals")


################################################################################
## FIT AUTO ARIMA MODEL.
# Use auto.arima() function to fit ARIMA model.
# Use summary() to show auto ARIMA model and its parameters.
train.auto.arima <- auto.arima(train.ts)
summary(train.auto.arima)

# Apply forecast() function to make predictions for ts with 
# auto ARIMA model in validation set.  
train.auto.arima.pred <- forecast(train.auto.arima, h = nValid, level = 0)
train.auto.arima.pred

# Using Acf() function, create autocorrelation chart of auto ARIMA 
# model residuals.
Acf(train.auto.arima$residuals, lag.max = 12, main = "Autocorrelations of Auto ARIMA Model Residuals")


################################################################################
# Use accuracy() function to identify common accuracy measures 
# for validation period forecast:
# (1) AR(2) model; 
# (2) MA(2) model; 
# (3) ARMA(2,2) model; 
# (4) ARIMA(2,1,2) model; 
# (5) ARIMA(2,1,2)(1,1,2) model; and 
# (6) Auto ARIMA model.
round(accuracy(train.ar2.pred$mean, valid.ts), 3)
round(accuracy(train.ma2.pred$mean, valid.ts), 3)
round(accuracy(train.arma2.pred$mean, valid.ts), 3)
round(accuracy(train.arima.pred$mean, valid.ts), 3)
round(accuracy(train.arima.seas.pred$mean, valid.ts), 3)
round(accuracy(train.auto.arima.pred$mean, valid.ts), 3)

################################################################################
## FIT SEASONAL ARIMA AND AUTO ARIMA MODELS FOR ENTIRE DATA SET. 
## FORECAST AND PLOT DATA, AND MEASURE ACCURACY.
# Use arima() function to fit seasonal ARIMA(2,1,2)(1,1,2) model 
# for entire data set.
# use summary() to show auto ARIMA model and its parameters for entire data set.
arima.seas <- Arima(price_per_galon_arima.ts, order = c(2,1,2), seasonal = c(1,1,2)) 
summary(arima.seas)

# Apply forecast() function to make predictions for ts with 
# seasonal ARIMA model for the future 12 periods. 
arima.seas.pred <- forecast(arima.seas, h = 12, level = 0)
arima.seas.pred

# Use Acf() function to create autocorrelation chart of seasonal ARIMA 
# model residuals.
Acf(arima.seas$residuals, lag.max = 12, main = "Autocorrelations of Seasonal ARIMA (2,1,2)(1,1,2) Model Residuals")

################################################################################
# Use auto.arima() function to fit ARIMA model for entire data set.
# use summary() to show auto ARIMA model and its parameters for entire data set.
auto.arima <- auto.arima(price_per_galon_arima.ts)
summary(auto.arima)

# Apply forecast() function to make predictions for ts with 
# auto ARIMA model for the future 12 periods. 
auto.arima.pred <- forecast(auto.arima, h = 12, level = 0)
auto.arima.pred

# Use Acf() function to create autocorrelation chart of auto ARIMA 
# model residuals.
Acf(auto.arima$residuals, lag.max = 12, main = "Autocorrelations of Auto ARIMA Model Residuals")


################################################################################
# Use accuracy() function to identify common accuracy measures for:
# (1) Seasonal ARIMA (2,1,2)(1,1,2) Model,
# (2) Auto ARIMA Model,
# (3) Seasonal naive forecast, and
# (4) Naive forecast.
round(accuracy(arima.seas.pred$fitted, price_per_galon_arima.ts), 3)
round(accuracy(auto.arima.pred$fitted, price_per_galon_arima.ts), 3)
round(accuracy((snaive(price_per_galon_arima.ts))$fitted, price_per_galon_arima.ts), 3)
round(accuracy((naive(price_per_galon0_arima.ts))$fitted, price_per_galon_arima.ts), 3)

################################################################################################################################################################














































