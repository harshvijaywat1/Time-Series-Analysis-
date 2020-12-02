// Databricks notebook source
// DBTITLE 1,Time Series Analysis of dataset downloaded from https://www.kaggle.com/kanncaa1/time-series-prediction-tutorial-with-eda
//Time Series Analysis of dataset downloaded from https://www.kaggle.com/kanncaa1/time-series-prediction-tutorial-with-eda

// COMMAND ----------

//We have used 2 libraries for this, i.e. sparkts and flint

// COMMAND ----------

/*1. Load the Data
 
 It contains 2 files:
i) Aerial Bombing Operations in WW2
ii) Weather Conditions in WW2(This data set has 2 subset in it. First one includes weather station locations like country, latitude and longitude.
Second one includes measured min, max and mean temperatures from weather stations.)
*/



// COMMAND ----------

//importing sparkts and required java time classes 
import java.time.{LocalDateTime, ZoneId, ZonedDateTime}

import com.cloudera.sparkts._
import com.cloudera.sparkts.stats.TimeSeriesStatisticalTests

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

//bombing data
var aerial = spark.read.
                    format("csv")
                    .option("header","true")
                    .option("inferSchema","true")
                    .option("timestampFormat","MM/dd/yyyy")
                    .option("mode", "PERMISSIVE")
                    .load("dbfs:/FileStore/shared_uploads/harsh.vijaywat@gmail.com/operations.csv") 
//weather data that includes locations like country, latitude and longitude.
var weather_station_locations = spark.read.
                    format("csv")
                    .option("header","true")
                    .option("inferSchema","true")
                    .option("timestampFormat","MM/dd/yyyy")
                    .option("mode", "PERMISSIVE")
                    .load("dbfs:/FileStore/shared_uploads/harsh.vijaywat@gmail.com/Weather_Station_Locations.csv")        

val weatherSchema = StructType(Array(StructField("STA", StringType), StructField("Date", TimestampType), StructField("MeanTemp", DoubleType)))
//Second weather data that includes measured min, max and mean temperatures
var weather = spark.read.
                    format("csv")
                    .option("header","true")
                    .schema(weatherSchema)
                    .option("timestampFormat","yyyy/MM/dd")
                    .option("mode", "PERMISSIVE")
                    .load("dbfs:/FileStore/shared_uploads/harsh.vijaywat@gmail.com/Summary_of_Weather.csv")        




// COMMAND ----------

/*Data Description

1. Aerial bombing Data description:
Mission Date: Date of mission
Theater of Operations: Region in which active military operations are in progress; "the army was in the field awaiting action"; Example: "he served in the Vietnam theater for three years"
Country: Country that makes mission or operation like USA
Air Force: Name or id of air force unity like 5AF
Aircraft Series: Model or type of aircraft like B24
Callsign: Before bomb attack, message, code, announcement, or tune that is broadcast by radio.
Takeoff Base: Takeoff airport name like Ponte Olivo Airfield
Takeoff Location: takeoff region Sicily
Takeoff Latitude: Latitude of takeoff region
Takeoff Longitude: Longitude of takeoff region
Target Country: Target country like Germany
Target City: Target city like Berlin
Target Type: Type of target like city area
Target Industry: Target industy like town or urban
Target Priority: Target priority like 1 (most)
Target Latitude: Latitude of target
Target Longitude: Longitude of target

2. Weather Condition data description:
Weather station location:
WBAN: Weather station number
NAME: weather station name
STATE/COUNTRY ID: acronym of countries
Latitude: Latitude of weather station
Longitude: Longitude of weather station

3. Weather:
STA: eather station number (WBAN)
Date: Date of temperature measurement
MeanTemp: Mean temperature */

// COMMAND ----------

/* Data Cleaning

Aerial Bombing data includes a lot of NaN value. Instead of using them,  drop some NaN values. It does not only remove the uncertainty but it also ease visualization process.
Drop countries that are NaN
Drop if target longitude is NaN
Drop if takeoff longitude is NaN
Drop unused features
*/

// COMMAND ----------

//drop if target longitude, takeoff longitude ,Country are NaN
aerial = aerial.na.drop("any", Array("Country", "Target Latitude", "Takeoff Longitude"))
//drop unused features
aerial = aerial.drop("Mission ID", "Unit ID", "Target ID", "Altitude (Hundreds of Feet)", "Attacking Aircraft", "Bombing Aircraft", "Aircraft Returned", "Aircraft Failed", "Aircraft Damaged", "Aircraft Lost", "High Explosives", "High Explosives Type", "Mission Type", "High Explosives Weight (Pounds)", "High Explosives Weight (Tons)", "Incendiary Devices", "Incendiary Devices Type", "Incendiary Devices Weight (Pounds)", "Incendiary Devices Weight (Tons)", "Fragmentation Devices", "Fragmentation Devices Type", "Fragmentation Devices Weight (Pounds)", "Fragmentation Devices Weight (Tons)", "Total Weight (Pounds)", "Total Weight (Tons)", "Time Over Target", "Bomb Damage Assessment", "Source ID" )
//drop  takeoff latitude=4248, takeoff longitude=1355  
aerial = aerial.filter("`Takeoff Longitude` != 1355 AND `Takeoff Latitude` != 4248")
//select only important columns
weather_station_locations = weather_station_locations.select("WBAN", "NAME", "STATE/COUNTRY ID", "Latitude", "Longitude")
//select only important columns
weather = weather.select("STA", "Date", "MeanTemp")
//drop nan values in weather
weather = weather.na.drop

// COMMAND ----------

//Data Visualization

// COMMAND ----------

//How many country which attacks
display(aerial.groupBy("Country").count())

// COMMAND ----------

//Top target countries
display(aerial.groupBy("Target Country").count())

// COMMAND ----------

//Top 10 aircraft series
display(aerial.groupBy("Aircraft Series").count().orderBy("Aircraft Series"))

// COMMAND ----------

//MeanTemp as function of date
display(weather.select("Date", "MeanTemp"))

// COMMAND ----------

weather.show(10)

// COMMAND ----------

//Time Series Prediction with ARIMA

// COMMAND ----------

/*Stationarity of a Time Series

There are three basic criterion for a time series to understand whether it is stationary series or not.
Statistical properties of time series such as mean, variance should remain constant over time to call time series is stationary
constant mean
constant variance
autocovariance that does not depend on time. autocovariance is covariance between time series and lagged time series.

Lets visualize and check seasonality trend of our time series. */

// COMMAND ----------

 /*We can check stationarity using the following methods:

1.Plotting Rolling Statistics: We have a window lets say window size is 6 and then we find rolling mean and variance to check stationary.
2.Dickey-Fuller Test: The test results comprise of a Test Statistic and some Critical Values for difference confidence levels. If the test statistic is less than the critical value, we can say that time series is stationary. */

// COMMAND ----------

//importing flint packages/classes  for rolling statistics
import com.twosigma.flint.timeseries.TimeSeriesRDD
import scala.concurrent.duration._  // for defined value of DAYS
val tsRdd = TimeSeriesRDD.fromDF(weather)(isSorted = false, timeUnit = DAYS, timeColumn = "Date")

// COMMAND ----------

tsRdd.toDF.show(10)

// COMMAND ----------

//Calculation of rolling mean and rolling std of MeanTemp
import com.twosigma.flint.timeseries._
val result = tsRdd.summarizeWindows(
 Windows.pastAbsoluteTime("6days"),
 Summarizers.mean("MeanTemp")).summarizeWindows(
 Windows.pastAbsoluteTime("6days"),
 Summarizers.stddev("MeanTemp"))


// COMMAND ----------

//visualtisation of rolling stats
display(result.toDF)

// COMMAND ----------

/*Our first criteria for stationary is constant mean. So we fail because mean is not constant as you can see from plot(black line) above . (no stationary)
Second one is constant variance. We fail this too because std  is not constant.
As a result, we sure that our time series is not stationary.
Lets make time series stationary at the next part.
*/


// COMMAND ----------

result.toDF.show(10)

// COMMAND ----------

/*Make a Time Series Stationary?

First solve trend(constant mean) problem
Most popular method is moving average.
Moving average: We have window that take the average over the past 'n' sample. 'n' is window size.


// COMMAND ----------

//Moving average method
val o = result.toDF.withColumn("Moving_avg_diff", col("MeanTemp")-col("MeanTemp_mean"))
val tsRdd2 = TimeSeriesRDD.fromDF(o)(isSorted = false, timeUnit = DAYS, timeColumn = "time")

// COMMAND ----------

val result2 = tsRdd2.summarizeWindows(
 Windows.pastAbsoluteTime("6days"),
 Summarizers.mean("Moving_avg_diff")).summarizeWindows(
 Windows.pastAbsoluteTime("6days"),
 Summarizers.stddev("Moving_avg_diff"))

// COMMAND ----------

display(result2.toDF.select("time","Moving_avg_diff","Moving_avg_diff_mean","Moving_avg_diff_stddev"))
//Constant mean criteria: mean looks like constant as you can see from plot(orange line) above . (yes stationary)

// COMMAND ----------

/*Forecasting a Time Series

For prediction(forecasting) we will use ts_diff time series that is result of moving avg method. 
Also prediction method is ARIMA that is Auto-Regressive Integrated Moving Averages.
AR: Auto-Regressive (p): AR terms are just lags of dependent variable. For example lets say p is 3, we will use x(t-1), x(t-2) and x(t-3) to predict x(t)
I: Integrated (d): These are the number of nonseasonal differences. For example, in our case we take the first order difference. So we pass that variable and put d=0
MA: Moving Averages (q): MA terms are lagged forecast errors in prediction equation.
(p,d,q) is parameters of ARIMA model.
In order to choose p,d,q parameters we will use two different plots.
Autocorrelation Function (ACF): Measurement of the correlation between time series and lagged version of time series.
Partial Autocorrelation Function (PACF): This measures the correlation between the time series and lagged version of time series but after eliminating the variations already explained by the intervening comparisons.*/

// COMMAND ----------

//we need array of double to calculate acf and pacf (due to scala APIs)
val cc = result2.toDF.select("Moving_avg_diff").rdd.collect.map(_.getDouble(0))

// COMMAND ----------

//importing required dependencies
import org.apache.commons.math3.distribution.NormalDistribution
import com.cloudera.sparkts.models.Autoregression
import org.apache.spark.mllib.linalg._
//method for finding confident interval value (same for +ve and -ve)
def calcConfVal(conf: Double, n: Int): Double = {
    val stdNormDist = new NormalDistribution(0, 1)
    val pVal = (1 - conf) / 2.0
    stdNormDist.inverseCumulativeProbability(1 - pVal) / Math.sqrt(n)
  }

 

// COMMAND ----------

def acf(data: Array[Double], maxLag: Int, conf: Double = 0.95) = {
    // calculate correlations and confidence bound
    val autoCorrs = UnivariateTimeSeries.autocorr(data, maxLag)
    val confVal = calcConfVal(conf, data.length)
 }

 def pacf(data: Array[Double], maxLag: Int, conf: Double = 0.95) = {
    // create AR(maxLag) model, retrieve coefficients and calculate confidence bound
    val model = Autoregression.fitModel(new DenseVector(data), maxLag)
    val pCorrs = model.coefficients // partial autocorrelations are the coefficients in AR(n) model
    val confVal = calcConfVal(conf, data.length)
 }



// COMMAND ----------

//calculating acf 
val autoC = UnivariateTimeSeries.autocorr(cc, 20)

// COMMAND ----------

//calculating confident interval value for 5% 
val confVal2 = calcConfVal(0.95, cc.length)

// COMMAND ----------

//case class for reconverting Array[Double] to Dataframe for visualisation
case class acfclass(x:Double, corr:Double, conf:Double)  // x is lag value, corr is autocorrelation, conf is confidence interval value 

// COMMAND ----------

//collection of acfclass objects
val f = for(i <- 0 to 19)yield{                 
  acfclass(i, autoC(i) ,confVal2 )                 
  
}

// COMMAND ----------

//plot acf graph
display(f.toSeq.toDF)

// COMMAND ----------

//acf graph crosses the confident line for the first time at lag value x= 13, therefore q=13 

// COMMAND ----------

//calculating pacf
// create AR(maxLag) model, retrieve coefficients and calculate confidence bound 
val model = Autoregression.fitModel(new DenseVector(cc), 20)
    val pCorrs = model.coefficients // partial autocorrelations are the coefficients in AR(n) model

// COMMAND ----------

val f2 = for(i <- 0 to 19)yield{
  acfclass(i, pCorrs(i) ,confVal2 )                 
  
}

// COMMAND ----------

display(f2.toSeq.toDF)

// COMMAND ----------

//pacf graph crosses the confident line for the first time at lag value x= 5, therefore  p = 5

// COMMAND ----------

//Now lets use p=5, q=13, d=0  as parameters of ARIMA models and predict

// COMMAND ----------

//In scala, to apply ARIMA model we need vector therefore we convert column data to vector 
import org.apache.spark.mllib.linalg.Vector
import com.cloudera.sparkts.stats.TimeSeriesStatisticalTests
import org.apache.spark.mllib.linalg.Vectors 
val ts = Vectors.dense(weather.select("MeanTemp").rdd.collect.map(_.getDouble(0)))


// COMMAND ----------

//import required dependencies
import com.cloudera.sparkts.stats._
import com.cloudera.sparkts.models._
//fitting ARIMA model
val arimaModel = ARIMA.fitModel(5, 0, 13, ts)

// COMMAND ----------

//predicting 50 future values
arimaModel.forecast(ts, 50)

// COMMAND ----------

/*Conclusion
*We learn how to load and clean data.
*We learn how to visualize data.
*We learn how to make TimeSeriesRDD.
*We learn about stationarity, how to make a time series stationary.
*We learn how to make  plots with display function of databricks.
*We learn how to make time series forecast. */
