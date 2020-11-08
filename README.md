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

  <img width="650" height="300" src="https://github.com/kmair/covid-usa/blob/master/snaps/dashboard.png">
</p>

This project aims to track down the essential stats of COVID-19 within the US and estimating the cases and deaths based on time-series analysis and deep learning

Track the latest developments of Covid in the US [here](https://usa-covid19-forecasts.herokuapp.com/)

## Overview

The project accomplishes 2 major things:

**Important stats**

**Future forecasts**

![forecasts](https://github.com/kmair/covid-usa/blob/master/snaps/prediction.gif)
Based on the state selected on the map, the forecast for 10 days in future is displayed. The models running in the background are:
- ARIMA model
- LSTM model

## Usage

To reproduce this project locally, do the following steps:
1. Clone the repository
2. Setup the environment
3. Run the `app.py` file


## Contributing

To provide additional inputs to the dashboard or make changes, feel free to send a pull request. Some of the aspects one can look at are:
1. Modeling the time series county-wise for more discrete stats and forecasts.
The major issue would be to create county-wise forecasts owing to the high variance in cases across counties making the forecasts extremely unpredictable and even stationarity of the model for each and every county makes it very difficult to analyze and normalize before predicting. 
To check 

## Project Status 

The project is currently deployed on Heroku and can be cloned locally on the device by setting up the environment using the `requirements.txt` file
