# Wind energy forecasting 

## Context
<p>
This project aims at modeling and forecasting wind energy generation for a wind farm in Western Denmark Horns Rev with a nominal capacity of 160MW, while the weather forecast input information is from the European Centre for Medium-range Weather Forecasts (ECMWF - ecmwf.int), the world-leading research and operational weather forecasting centre.
</p>

The project is for a wind-forecasting competition held at Technical University of Denmark and inspired by the recent Global Energy Forecasting Competitions held in 2012 and 2014 held on <a url=https://www.kaggle.com>kaggle.com</a> with 50 teams. The competition is organized in 4 stages.

## Introduction

Wind power forecasts are essential inputs to a number of decision-making problems in electricity markets. It is therefore a good idea to understand how these forecasts can be generated. You will have to generate wind power forecasts for given periods, with weather forecasts as input, and with a long history of past cases (2 years) to learn from. Such historical data includes the weather forecasts themselves, and the power observations that were eventually collected at the wind farm. Therefore, what can be learned from these data is the relationship between weather forecasts and the power eventually produced at the wind farm. Based on this relationship learned from the past, one is to apply it to input weather forecasts to issue genuine wind power forecasts over the evaluation periods defined for the various stages of the forecast competition.
In the present case, the input data consists in wind forecasts at 2 heights (10m and 100m above ground level). Wind forecasts are given in terms of their zonal and meridional components (u and v), which correspond to the projection of the wind vector on West-East and South-North axes, respectively. Weather forecasts are issued once a day at 00:00, while they have lead times between 1 and 24 hours ahead