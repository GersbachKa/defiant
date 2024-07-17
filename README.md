# defiant 

DEFIANT (Data-driven Enhanced Frequentist Inference Analysis with Next-gen Techniques) is a python package primarily aiming for robust, fast, and frequentist analysis of Pulsar Timing Array data using the PTA Optimal Statistic (OS). This package presents the parallel pipeline to the Bayesian PTA pipeline [ENTERPRISE](https://github.com/nanograv/enterprise). 

This package is primarily based on the work of the [PTA Optimal Statistic](https://arxiv.org/abs/0809.0701) but encorporates many of the expansions on the method developed since then. For details on what OS expansions this package supports, check the [Usage](#usage) section.

## Installation

TODO: Make this package installable...

For now, you can copy the local location of your 'defiant' folder and add it to system path.
You can do this in python with the following command

```python
import sys
sys.path.append('<path-to-defiant>/defiant')
from defiant import OptimalStatistic
```


## Usage

There are many ways in which you can use DEFIANT. Most commonly, folks want to run the Optimal Statistic. This can be found in [defiant.OptimalStatistic](https://github.com/GersbachKa/defiant/blob/main/defiant/core.py) (or equivalently [defiant.core.OptimalStatistic](https://github.com/GersbachKa/defiant/blob/main/defiant/core.py)).

To use the OptimalStatistic, you must instance the object. There are 3 required parameters: 
1. '[psrs](https://github.com/GersbachKa/defiant/blob/main/defiant/core.py#L45)' -> The list of [enterprise.Pulsar]() objects
2. '[pta](https://github.com/GersbachKa/defiant/blob/main/defiant/core.py#L45)' -> The [enterprise.signals.signal_base.PTA]() object you wish to analyze 
3. '[gwb_name](https://github.com/GersbachKa/defiant/blob/main/defiant/core.py#L45)' -> A string corresponding to the name of the GWB parameters in the PTA.param_names*

*note that defiant assumes that you create PTAs in such a way that the parameters corresponding to the GWB are have names like `gwb_name + "_log10_A"` and `gwb_name + "_log10_A"`. By default '[gwb_name](https://github.com/GersbachKa/defiant/blob/main/defiant/core.py#L45)' is set to `"gw"`.

```python
# After defining psrs and pta
from defiant import OptimalStatistic
OS_obj = OptimalStatistic(psrs, pta, gwb_name='gw')
```

There are many optional parameters upon instancing which you can use to skip some additional steps, but for simplicity, we will show it in full. You will get a warning if you do not supply the Bayesian Common Uncorrelated Red Noise (CURN) MCMC chain in the initialization, but we can set it afterwards using the [OS_obj.set_chain_params()]() method. This method takes in parameters in many varieties of ways. Checking the documentation is the best way to find your favorite method. The easiest way is to supply a pre-made [la_forge.core.Core](https://github.com/nanograv/la_forge) object (check la_forge [documentation](https://la-forge.readthedocs.io/en/latest/tutorial1.html) for how to make a core from chain files). 

```python
# After instancing a la_forge core named lfcore
OS_obj.set_chain_params(core=lfcore)
```

Next and finally, we need to decide on which version of the Optimal Statistic you run. There are many options and they can all work together. If you want a fun, interactive way to figure out which version to run, check the [defiant.fun.what_kind_of_OS_are_you()](https://github.com/GersbachKa/defiant/blob/main/defiant/fun.py) function! If you'd rather a less silly way, you can check the DEFIANT choice tree below:

![image](defiant_OS_choice_tree.png)

Once you've decided on an OS version to run, check how to run that code using the choice tree or [what_kind_of_OS_are_you()](https://github.com/GersbachKa/defiant/blob/main/defiant/fun.py) and run it! There are a few options that you might want to be aware of, specifically `return_pair_vals` which, if set to false, will not return pair-wise correlated values (the intermediate data products of the OS). This can be helpful in cases where you use noise marginalization and pair covariance as storing each covariance matrix can be memory expensive.