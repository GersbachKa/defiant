
import numpy as np


def get_MDC1_psrs(use_pickle=False):
    """A function to grab the MDC1 dataset and return it as a list of Enterprise Pulsar objects.

    NOTE: If you do not trust me, you should set use_pickle to False and load the pulsars
    from the par and tim files as pickle files can contain malicous code. If you do
    trust me, using the pickle file will be much faster and will crash significantly
    less.

    Args:
        use_pickle (bool, optional): Whether to use the pickle file of the MDC1 pulsars.

    Returns:
        list, dict: A list of MDC1 pulsars and a dictionary of the injected parameters.
    """
    if use_pickle:
        import pickle
        pickle_loc = __file__[:-13]+'mdc1_psrs.pkl' # Remove this files name, add pickle
        with open(pickle_loc,'rb') as f:
            psrs = pickle.load(f)
    
    else:
        import enterprise
        from enterprise.pulsar import Pulsar
        from glob import glob
        from tqdm.auto import tqdm

        datadir = enterprise.__path__[0] + '/datafiles/mdc_open1/'
        parfiles = sorted(glob(datadir + '/*.par'))
        timfiles = sorted(glob(datadir + '/*.tim'))


        psrs = []
        for i in tqdm(range(len(parfiles)),desc='Loading MDC1 psrs'):
            psr = Pulsar(parfiles[i],timfiles[i],)
            psrs.append(psr)
            del psr
    
    inj_params = {'gw_log10_A':np.log10(5e-14),'gw_gamma':(13./3.)}

    return psrs, inj_params


def create_MDC1_like_psrs(gwb_amplitude=5e-14, gwb_gamma=13./3., 
                          irn_amplitude=None, irn_gamma=13./3., irn_components=30,
                          toaerr=0.1, tspan=5.0, nfit=1, seed=None):
    """A function to create MDC1-like simulated datasets using user-specified parameters.

    This function will create a list of pulsars with observations once every 2 weeks 
    for the number of years specified using Libstempo's toasim module. The pulsars 
    will have a gravitational wave background (GWB) injected with the specified 
    amplitude and spectral index. Intrinsic red noise (IRN) can be injected into 
    each pulsar with the specified amplitude and spectral index. The intrinsic red 
    noise can either be set to the same value in all pulsars (by giving 
    irn_amplitude and irn_gamma single values) or set to different values in 
    each pulsar (by supplying arrays for irn_amplitude and irn_gamma). Additionally, 
    the white noise will be set such that there is only EFAC set at a constant 1 
    for all pulsars.

    Args:
        gwb_amplitude (float): The amplitude of the GWB (NOT LOG). Defaults to 5e-14.
        gwb_gamma (float): The spectral index of the GWB. Defaults to 13/3.
        irn_amplitude (float or np.ndarray, optional): The intrinsic red noise 
            amplitude(s). Set to None for no IRN. Defaults to None.
        irn_gamma (float or np.ndarray, optional): The spectral index(s) for the 
            intrinsic red noise in each pulsar. Unused if irn_amplitude is None.
        irn_components (int, optional): The number of Fourier components to simulate
            the IRN for. Defaults to 50.
        toaerr (float, optional): The error in all TOAs [in microseconds]. Defaults to 0.1.
        tspan (float, optional): The number of years the dataset contains. Defaults to 5. 
            Note that this also changes how many TOAs are in the dataset.
        nfit (int, optional): The number of iterative timing model fits for each pulsar. 
            Can result in some crashing if set high. Defaults to 1.
        seed (int, optional): The seed for a particular simulation. Set to None for
            a random seed. Defaults to None.

    Raises:
        Exception: If the pulsar fit fails due to phase wrapping
        Exception: If the phase wrapping still exists after 3 aditional fit attempts

    Returns:
        list, dict: A list of the simulated pulsars and a dictionary of the injected parameters
    """

    import enterprise
    from enterprise.pulsar import Pulsar
    from glob import glob

    from libstempo import toasim as LT
    from tqdm.auto import tqdm

    inj_params={}
    seed = int(np.random.random()*2**31-1) if seed==None else int(seed)

    datadir = enterprise.__path__[0] + '/datafiles/mdc_open1/'
    parfiles = sorted(glob(datadir + '/*.par'))

    obstimes = 53000.0 + np.linspace(0.0,tspan,int(26*tspan)) * 365.25

    ltpsrs = []
    for i in tqdm(range(len(parfiles)),desc='Creating psrs'):
        p = parfiles[i]
        psr = LT.fakepulsar(parfile=p, obstimes=obstimes, toaerr=toaerr)
        LT.add_efac(psr,efac=1.0,seed=seed)
        seed+=1

        if irn_amplitude is not None and irn_gamma is not None:
            if np.array(irn_amplitude).size>1 and np.array(irn_gamma).size>1:
                ia,ig = irn_amplitude[i],irn_gamma[i]
            else:
                ia,ig = irn_amplitude,irn_gamma
            
            LT.add_rednoise(psr,ia,ig,components=irn_components,seed=seed)
            seed+=1

            inj_params[f'{psr.name}_red_noise_gamma'] = irn_gamma
            inj_params[f'{psr.name}_red_noise_log10_A'] = np.log10(irn_amplitude)

        ltpsrs.append(psr)
        
    
    LT.createGWB(ltpsrs,Amp=gwb_amplitude,gam=gwb_gamma,seed=seed)
    inj_params['gw_log10_A']=np.log10(gwb_amplitude)
    inj_params['gw_gamma']=gwb_gamma
    seed+=1

    for i,p in enumerate(ltpsrs):
        try:
            p.fit(nfit)
        except:
            print(f'Pulsar fit failed on {p.name}, [index {i}], skipping...')


    # Find phase wrapings!
    for i,p in enumerate(ltpsrs):
        fit_attempts = 0
        while _has_phase_wrapping(p):
            # Has phase wrapping, try to fit it away!
            try:
                p.fit()
            except:
                msg = f'Pulsar fit failed while containing phase wraps [{p.name}]. '+\
                       'Data will be flawed.'
                raise Exception(msg)
            fit_attempts+=1
            if fit_attempts>3:
                msg = f'Pulsar fit failed while containing phase wraps after 3 attempts [{p.name}]. '+\
                       'Data will be flawed.'
                raise Exception(msg)


    psrs = [Pulsar(p) for p in ltpsrs]
    
    return psrs, inj_params


def _has_phase_wrapping(ltpsr):
    """A simple function to check if a pulsar's residuals wrap in phase.

    This function checks if any two adjacent residuals have differences on the
    same order of magnitude as the pulsar's period. While this method may produce
    false positives if there are large gaps between residuals, it works fairly 
    well for MDC1-like datasets.

    Args:
        ltpsr (LT.fakepulsar): A libstempo fakepulsar object

    Returns:
        bool: Whether there is likely phase wrapping
    """
    diffs = np.abs(np.diff(ltpsr.residuals()))
    P0 = 1/(ltpsr.vals()[ltpsr.pars().index('F0')])
        
    return max(diffs)>(P0/10)



def create_MDC1_PTA(psrs, gwb_psd='powerlaw', gwb_components=10, gwb_gamma=13./3., 
                    orf='crn', gwb_name='gw', include_irn=False, irn_components=30, 
                    use_marginalizingtm=True):
    """A function to easily create MDC1-like PTA models.

    A function used to easily create MDC1-like PTA models. Importantly, EFAC is 
    always set to 1 with non-varying white noise. The include_irn controls whether 
    intrinsic red noise is added to the pulsar models.

    Args:
        psrs (list): A list of Enterprise Pulsars to apply the resulting model to
        gwb_psd (str): The PSD model of the GWB. Defaults to 'powerlaw'.
            Must be either 'powerlaw' or 'spectrum'.
        gwb_components (int): The number of GWB frequencies to analyze. Defaults to 10.
        gwb_gamma (float, optional): The spectral index of a powerlaw model. Can be
            set to None for a varied gamma model. Defaults to 13/3.
        orf (str): The ORF model to apply to the PTA. Defaults to 'crn' (common red noise).
            Must be an ORF model built into enterprise_extensions.
        gwb_name (str, optional): The name of the GWB model component. Defaults to 'gw'.
        include_irn (bool): Whether to include IRN searching. Defaults to False.
        irn_components (int, optional): If include_irn is True, the number of IRN
            components to search over. Defaults to 30.
        use_marginalizingtm (bool, optional): Whether to use a more accuratge, but 
            previously Optimal Statistic incompatible model (defiant.OptimalStatistic) 
            does support the MarginalizingTimingModel). Defaults to True.

    Returns:
        pta: The enterprise.signals.signal_base.PTA object containing the model specified
    """
    
    from enterprise_extensions.model_utils import get_tspan
    from enterprise.signals import parameter,white_signals,gp_signals,signal_base

    from enterprise_extensions import blocks


    Tspan = get_tspan(psrs)

    efac = parameter.Constant(1.0)
    ef = white_signals.MeasurementNoise(efac=efac)

    if gwb_psd == 'spectrum':
        gwb = blocks.common_red_noise_block(psd=gwb_psd,Tspan=Tspan,components=gwb_components, orf=orf,
                                             gamma_val=gwb_gamma,logmin=-18,logmax=-1,name=gwb_name)
    else:
        gwb = blocks.common_red_noise_block(psd=gwb_psd,Tspan=Tspan,components=gwb_components, orf=orf,
                                             gamma_val=gwb_gamma,logmin=-18,logmax=-12,name=gwb_name)

    irn = blocks.red_noise_block(psd='powerlaw',Tspan=Tspan,components=irn_components)

    if use_marginalizingtm:
        tm = gp_signals.MarginalizingTimingModel()
    else:
        tm = gp_signals.TimingModel(use_svd=True)

    if include_irn:
        model = tm + ef + gwb + irn
    else:
        model = tm + ef + gwb

    pta = signal_base.PTA([model(psr) for psr in psrs])
    return pta


