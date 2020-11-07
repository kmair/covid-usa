# Covid-19 in USA

<p align="center">
  <a href=https://github.com/kmair/covid-usa/blob/master/LICENSE>
    <img src="https://img.shields.io/apm/l/atomic-design-ui.svg?">
  </a>
  <a>
    <img src="https://img.shields.io/pypi/pyversions/yt2mp3.svg">
  </a>

  <a>
    <img src="http://img.shields.io/badge/Status-Active-green.svg">
  </a>
  <a>
    <img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
  </a>
</p>

This is an effort to track down the essential stats of COVID-19 within the US and estimating the cases and deaths based on time-series analysis and deep learning

Track the latest developments of Covid in the US [here](https://usa-covid19-forecasts.herokuapp.com/)



## Contributing

To provide additional inputs to the dashboard or make changes, feel free to send a pull request. Some of the aspects one can look at are:
1. Modeling the time series county-wise for more discrete stats and forecasts.
The major issue would be to create county-wise forecasts owing to the high variance in cases across counties making the forecasts extremely unpredictable and even stationarity of the model for each and every county makes it very difficult to analyze and normalize before predicting. 
To check 

## Project Status 

The project is currently deployed on Heroku and can be cloned locally on the device by setting up the environment using the `requirements.txt` file
