<h1><span style="color:red"> Wind energy forecasting </span></h1>

## Context
<p>
This project aims at modeling and forecasting wind energy generation over an evaluation periods for a wind farm in Western Denmark Horns Rev with a nominal capacity of 160MW, while the weather forecast input information is from the European Centre for Medium-range Weather Forecasts (ECMWF - ecmwf.int), the world-leading research and operational weather forecasting centre.
</p>

The project is for a wind-forecasting competition held at Technical University of Denmark and inspired by the recent Global Energy Forecasting Competitions held in 2012 and 2014 held on <a url=https://www.kaggle.com>kaggle.com</a> with 50 teams. The competition is organized in 4 stages.

## Introduction


Wind power forecasts are essential inputs to a number of decision-making problems in electricity markets. Understanding how we can generate these forecasts is a great asset to making better revenue-maximization trading strategies. Trading startegies can adopt two approaches:

* Deterministic approach: We generate point predictions of wind power generation (persistence, climatology or with an advanced model) which determine our bids in spot markets

* Probabilistic approach: We generate probabilistic forecasts such as quantiles, intervals and predictive distributions of wind power generation with a model of the participant’s sensitivity to regulation costs.


## Available Data

Historical Data is provided on which the models are to be trained. 
It includes the weather forecasts themselves, and the power observations that were eventually collected at the wind farm. Therefore, what can be learned from these data is the relationship between weather forecasts and the power eventually produced at the wind farm.

In the present case, the input data consists in wind forecasts at 2 heights (10m and 100m above ground level). Wind forecasts are given in terms of their zonal and meridional components (u and v), which correspond to the projection of the wind vector on West-East and South-North axes, respectively. Weather forecasts are issued once a day at 00:00, while they have lead times between 1 and 24 hours ahead


## Approach

We conduct three different approaches to this problem with different levels of complexity

* ** Persistence / Climatology ** 


* ** Regression models (Lasso, Random Forest etc) ** 


* ** Time series models (ARIMA) **

