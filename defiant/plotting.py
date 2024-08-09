
from matplotlib import pyplot as plt
from scipy.stats import binom
import numpy as np
from . import utils
from .orf_functions import get_orf_function


def create_correlation_plot(xi,rho,sig,C,A2,A2s,bins=10,orf=['hd'],eig_thresh=1e-10,
                            **subplot_kwargs):
    """A function to create a pulsar separation vs correlation plot

    This function creates correlation plots for an OptimalStatistic (OS) run. This 
    function only supports single OS iterations (i.e. no noise marginalization) and
    only supports compute_OS() [or alternatively compute_PFOS() if given a single
    frequency's rho, sig, C, A2, and A2s]. This function DOES support the multi-component
    optimal statistic and pair covariance, however, if using both, the binned correlation
    estimators will only be calculated using the first element of orf. This function
    also only works for pre-implemented ORFs in defiant.orf_functions.defined_orfs

    The list of pre-defined orfs are:
        - 'hd' or 'hellingsdowns': Hellings and Downs
        - 'dp' or 'dipole': Dipole
        - 'mp' or 'monopole': Monopole
        - 'gwdp' or 'gw_dipole': Gravitational wave dipole
        - 'gwmp' or 'gw_monopole': Gravitational wave monopole
        - 'st' or 'scalar_tensor': Scalar tensor

    Args:
        xi (_type_): _description_
        rho (_type_): _description_
        sig (_type_): _description_
        C (_type_): _description_
        A2 (_type_): _description_
        A2s (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    if not hasattr(orf,'__iter__'):
        orf = [orf]

    fig, ax = plt.subplots(**subplot_kwargs)

    # Get bin averaged values
    if len(C.shape) > 2:
        # Pair covariant
        xia,rhoa,siga = utils.binned_pair_correlations(xi,rho,C,bins,orf)
    else:
        # Traditional
        xia,rhoa,siga = utils.binned_pair_correlations(xi,rho,sig,bins,orf)
    
    # Plot correlations
    ax.errorbar(xia,rhoa,siga,fmt='oC0',label='Binned Correlations',capsize=3)

    xi_range = np.linspace(0,np.pi,1002)[1:-1] # Avoid 0 and pi

    if len(A2)>1:
        # Multi-component
        means,mean_sig = utils.calculate_mean_sigma_for_MCOS(xi_range,A2,A2s,orf,clip_thresh=eig_thresh)
        # Plot the means
        ax.plot(xi_range, means, 'C1', label='$A^2$ Fit')
        ax.fill_between(xi_range, means-mean_sig, means+mean_sig, color='C1', alpha=0.1)

    else:
        # Single component
        orf_mod = get_orf_function(orf[0])(xi_range)
        ax.plot(xi_range, A2*orf_mod, 'C1', label='$A^2$ Fit')
        ax.fill_between(xi_range, (A2-A2s)*orf_mod, (A2+A2s)*orf_mod, color='C1', alpha=0.1)    

    ax.set_xlabel('Pulsar Separation (radians)')
    ax.set_ylabel('Correlated Power')
    ax.legend()
    plt.grid()
    plt.xlim(0,np.pi)
    return fig, ax
    




# P-P Plotting class -----------------------------------------------------------

class pp_plotter:
    def __init__(self, n_sims, n_steps=None, diag=True, square_aspect=True, **subplot_kwargs):
        """Instatiate a pp_plotter object

        Create a pp_plotter object. Useful for all sorts of pp plot needs!

        Args:
            n_sims (int): The number of percentiles (number of simulations)
            n_steps (int): The number of x positions for percentiles. Defaults to (1/5 * n_sims)
            diag (bool): Whether to plot as a diagonal or horizontal pp plot. Defaults to True.
            square_aspect (bool): Whether to make the plot square. Defaults to True.
            subplot_kwargs: Keyword arguments for pyplot's subplots function
        """
        
        self.fig, self.ax = plt.subplots(**subplot_kwargs)

        if square_aspect:
            self.ax.set_box_aspect(1)

        if n_steps==None:
            n_steps = int(n_sims/5)

        self.N = n_sims
        self.diag=diag
        self.n_steps = n_steps

        if self.diag:
            self.ax.set_ylabel('$P(\\theta < p)$')
            self.ax.set_xlabel('$p$')
            self.ax.set_xlim(0,1)
            self.ax.set_ylim(0,1)
        else:
            self.ax.set_ylabel('$P(\\theta < p) - p$')
            self.ax.set_xlabel('$p$')
            self.ax.set_xlim(0,1)
            
    
    def confidence_intervals(self, n_sigma=3, alpha=.1):
        """A function to add binomial distribution error bars to a PP plot

        This function was mostly taken from LIGO's Bilby software package:
        https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/result.py
        Specifically, make_pp_plot()

        Args:
            n_sigma (int): The number of sigma to plot [supports 1-4]. Defaults to 3.
            alpha (float): The alpha value for the sigma regions. Defaults to 0.1.
        """
        intervals = [0.68,0.95,0.997,0.999937]

        x_values = np.linspace(0,1,1001)
        for s,ci in zip(range(n_sigma),intervals):
            if self.diag:
                edge_of_bound = (1. - ci) / 2.0
                lower = binom.ppf(1 - edge_of_bound, self.N, x_values) / self.N
                upper = binom.ppf(edge_of_bound, self.N, x_values) / self.N

                lower[0] = 0
                upper[0] = 0
                lower[-1] = 1
                upper[-1] = 1
                self.ax.fill_between(x_values,lower,upper,alpha=alpha,color='k')
            else:
                nsigma = (s+1) * np.sqrt( x_values*(1-x_values)/ self.N )
                self.ax.fill_between(x_values,-nsigma,nsigma,alpha=alpha,color='k')


    def plot_percentile(self, injection, data, sigma=None, type='gaussian', format=None, 
                        **plot_kwargs):
        """A function to create the Percentile-Percentile plot. 

        This function will plot the Percentile-Percentile plot for a given set of 
        data and injection values. If the injection given is a scalar, it will be
        applied to all the datasets, otherwise the length of the injection must be
        the same as the length of the data. data consists of one of the following 3
        forms:
            1. 'gaussian': data consists of N mean values of the estimators and
                sigma is N uncertainties in those N estimators.
            2. 'distribution': data consists of N distributions of the estimators.
                This will calculate the percentile of the injection in each distribution.
                sigma is not used in this case. 
            3. 'noise_marginalized': Specific for the Noise Marginalized Optimal Statistic.
                data consists of N distributions of the estimators and sigma contains 
                N distributions of uncertainties in the estimators. This will use
                uncertainty sampling to find the percentile of in the full 
                estimator distribution.

        Finally, you can adjust the format of the plot through the format and plot_kwargs
        arguments.
        
        Args:
            injection (float or np.ndarray): The injected value(s) in data.
            data (np.ndarray): The N means or distributions to get percentiles from.
            sigma (np.ndarray): The N uncertainties in the data (if applicable).
            type (str): The type of data supplied. Must be 'gaussian', 'distribution', 
                or 'noise_marginalized'. Defaults to 'gaussian'.
            format (str): The format keyword of plt.plot(). Defaults to None.
            plot_kwargs: Keyword arguments for pyplot's plot function    
        """
        from scipy.stats import norm
        import numpy as np

        # Check if the injection is a scalar or some iterable
        if not hasattr(injection,'__iter__'):
            injection = np.ones( len(data) ) * injection
        elif len(data) != len(injection):
            raise Exception('Length of data does not match the length of injection')

        # Get percentiles
        perc = np.zeros(len(data))
        if type == 'gaussian':
            for i in range(len(data)):
                perc[i] = norm(loc=data[i],scale=sigma[i]).cdf(injection[i])
        elif type == 'distribution':
            for i in range(len(data)):
                dist = data[i]
                perc[i] = len( dist[ dist<=injection[i] ] ) / len(dist)
        elif type == 'noise_marginalized':
            for i in range(len(data)):
                dist = utils.uncertainty_sample(data[i],sigma[i])
                perc[i] = len( dist[ dist<=injection[i] ] ) / len(dist)

        # Get the x and y points
        x = np.linspace(0,1,self.n_steps)
        if self.diag:
            y = np.array( [np.sum([perc<=s])/len(perc) for s in x] )
        else:
            y = np.array( [np.sum([perc<=s])/len(perc) for s in x] ) - x

        if format   == None:
            self.ax.plot(x,y,**plot_kwargs)
        else:
            self.ax.plot(x,y,format,**plot_kwargs)

        return perc, x, y        


    def place_legend_outside(self):
        """A function to place the legend outside of the plot. 
        
        Handy because I can't remember how to place it outside...
        """
        self.ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')

