# PyTMLE

`PyTMLE` is a flexible Python implementation of the Targeted Maximum Likelihood Estimation (TMLE) framework for survival and competing risks outcomes. 

The package can be installed from PyPI, for example using `pip`:

```bash
pip install pytmle
```

It is designed to be easy to use with default models for initial estimates of nuisance functions which are applied in a super learner framework. With a `pandas` dataframe containing event times, indicators, and (binary) treatment group information in specified columns, it is straight-forward to fit a main `PyTMLE` class object and get predictions and plots for selected `target_times`:

```pytmle
from pytmle import PyTMLE

tmle = PyTMLE(df, 
              col_event_times="time", 
              col_event_indicator="status", 
              col_group="group", 
              target_times=target_times)

tmle.plot(type="risks") # get estimated counterfactual CIF, or set to "rr" or "rd" for ATE estimates based on RR or RD
pred = tmle.predict(type="risks") # store estimates in a data frame
```

However, it also allows for custom models to be used for the initial estimates or even passing initial estimates directly to the second TMLE stage.

Have a look at the package's [Read the Docs page](https://pytmle.readthedocs.io/) for the detailed API reference and tutorial notebooks.