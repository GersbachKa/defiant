from defiant import defiant_ascii, defiant_ascii_font_cred
import time

def what_kind_of_OS_are_you():
    """_summary_

    _extended_summary_
    """
    suggested_form = [None, None, None, None, 'Optimal Statistic']
    
    # Multiple ORF or single ORF
    print(defiant_ascii,'\n\n\n')
    print('Welcome to the next hottest gameshow in NANOGrav: What kind of OS are you? \n')
    print('In this game, we will determine what kind of Optimal Statistic you are based')
    print('on how you answer the following questions!','\n\n')
    
    print('Here we go!')
    print('Question 1: How many Overlap Reduction Functions (ORFs) are you using?')
    print('In other words, how many correlation patters are you searching for? [HD, Dipole, etc.]')
    norfs = int(input('Enter the (integer) number of ORFs you are using: '))

    if norfs > 1:
        print('Multiple ORFs means Multi-Component!')
        suggested_form[3] = 'Multi-Component'
    elif norfs == 1:
        print('Single ORF means Single-Component!')
    else:
        print('Invalid number of ORFs! Please try again.')
        return
    
    # Pair Covariance
    print('\n')
    print('Question 2: Would you estimate that your PTA Gravitational Wave Background (GWB)')
    print('signal is in the weak-signal (1), intermediate-signal (2), or strong-signal (3) regimes?')
    print('Type the number corresponding to your answer or 0 if you do not know.')
    strength = int(input('Enter the number (0,1,2,3): '))

    if strength==0:
        print('Unknown signal regime. If you do not know, we recommend using the pair covariance.')
        suggested_form[1] = 'Pair-Covariant'
    elif strength==1:
        print('Weak signal regime. Pair covariance is still most accurate, but you can')
        print('use the traditional form if you want to save time.')
        suggested_form[1] = 'Pair-Covariant'
    elif strength in [2,3]:
        print('Intermediate or strong signal regime. We recommend using the pair covariance.')
        suggested_form[1] = 'Pair-Covariant'
    else:
        print('Invalid signal regime! Please try again.')
        return
    
    # Noise marginalization
    print('\n')
    print('Question 3: Do you have a chain of MCMC parameters that you want to marginalize over?')
    marginalization = input('Enter "yes" or "no": ')

    if marginalization.lower() in ['yes','y']:
        print('Use noise marginalization!')
        suggested_form[2] = 'Noise-Marginalized'
    elif marginalization.lower() in ['no','n']:
        print('No need for noise marginalization!')
    else:
        print('Invalid input! Please try again.')
        return
    
    # Per frequency
    print('\n')
    print('Question 4: Are you trying to estimate for a broadband GWB or get individual')
    print('PSD estimates at each frequency? (1 for broadband, 2 for PSDs).')
    per_freq = int(input('Enter the number (1,2): '))

    if per_freq==1:
        print('For a broadband GWB, use the traditional form.')
    elif per_freq==2:
        print('(Individual frequency estimates)\n')
        print('Question 4.1: Does you test require complete frequency independence?')
        print('(not recommended for most cases)')
        freq_indep = input('Enter "yes" or "no": ')

        if freq_indep.lower() in ['yes','y']:
            print('Complete frequency independence uses the narrowband-normalized PFOS.')
            suggested_form[0] = 'narrowband-normalized Per-Frequency'
        elif freq_indep.lower() in ['no','n']:
            print('Use the broadband-normalized PFOS.')
            suggested_form[0] = 'broadband-normalized Per-Frequency'
        else:
            print('Invalid input! Please try again.')
            return
    else:
        print('Invalid input! Please try again.')
        return
    
    print('\n\n')
    print('Based on your answers, we find that the OS form that best suits you is: ')
    for i in range(3):
        time.sleep(1)
        print('.')
    
    final_form = ''        
    for f in suggested_form:
        if f is not None:
            final_form+=f+' '
    
    print('The',final_form,'\n\n')

    instructions = []
    code = []
    instructions.append('Now, how do we run this version?')
    instructions.append('First, create an DEFIANT OptimalStatistic object.')

    code.append('OS_obj = OptimalStatistic(psrs,pta=pta,gwb_name=\'gw\')')

    if suggested_form[2] is not None: # Noise Marginalization
        instructions.append('Next, we need to include an MCMC chain in the form of a la_forge.core.Core object.')
        code.append('OS_obj.set_chain_params(core=lfcore)')
        os_par = 'N=1000'
    else:
        os_par = 'params=params'

    
    if suggested_form[3] is not None: # Multi-Component
        instructions.append('After, with multiple ORFs, you need to set them with the set_orf() method.')
        code.append("OS_obj.set_orf(orf=['hd', 'dipole','monopole'])")
    else:
        instructions.append('After, with a single ORF, we need to set it with the set_orf() method.')
        code.append("OS_obj.set_orf(orf=['hd'])")

    if suggested_form[1] is not None:
        instructions.append('Next, when we call the compute_OS() or compute_PFOS() methods, the pair_covariance=True flag is default.')
        pc_par = 'pair_covariance=True'
    else:
        instructions.append('Next, when we call the compute_OS() or compute_PFOS() methods, the pair_covariance=False flag must be set.')
        pc_par = 'pair_covariance=False'
    
    if suggested_form[0] is not None: # Per-Frequency
        instructions.append('For Per-Frequency OS, we need to use the compute_PFOS() method.')
        if suggested_form[0] == 'broadband-normalized Per-Frequency':
            instructions.append('The broadband-normalized PFOS is the default [narrowband=False].')
            code.append('OS_obj.compute_PFOS('+os_par+','+pc_par+',narrowband=False)')
        else:
            instructions.append('The narrowband-normalized PFOS requires the narrowband=True flag.')
            code.append('OS_obj.compute_PFOS('+os_par+','+pc_par+',narrowband=True)')
    else:
        instructions.append('For the broadband OS (traditional OS), we use the compute_OS() method.')
        code.append('OS_obj.compute_os('+os_par+','+pc_par+')')

    instructions.append('The code for running this type of OS is:')
    for s in instructions:
        print(s)
    print('')
    for c in code:
        print(c)
