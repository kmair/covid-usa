# covid-usa
Analysis of the Covid in USA 

This is an effort to track down the essential stats of COVID-19 within the US and estimating the cases and deaths based on time-series analysis and deep learning

## Contributing

To provide additional inputs to the dashboard or make changes, feel free to send a pull request. Some of the aspects one can look at are:
1. Modeling the time series county-wise for more discrete stats and forecasts.
The major issue would be to create county-wise forecasts owing to the high variance in cases across counties making the forecasts extremely unpredictable and even stationarity of the model for each and every county makes it very difficult to analyze and normalize before predicting. 
To check 

## Project Status 

The project is currently being deployed on Heroku and can be replicated on the device by setting up the environment using the `requirements.txt` file