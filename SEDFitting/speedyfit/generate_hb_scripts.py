import h5py
import argparse
import glob
from astropy.table import Table,vstack
import numpy as np
import matplotlib.pyplot as plt

## SET THIS DIFFERENTLY IF DESIRED
EVI_EBV_factor = 1.25

def ext_index(obj):
    loc = obj[5:8]
    idx = int(obj[12:16])-1
    if loc == 'SMC':
        idx+=439
    return idx

def ext_str(obj,extTable,EVI_EBV_factor=1.25):
    idx = ext_index(obj)
    ebv = extTable['E(V-I)'][idx]/EVI_EBV_factor
    ebvm = extTable['-sigma1'][idx]/EVI_EBV_factor
    ebvp = extTable['+sigma2'][idx]/EVI_EBV_factor
    return f'\"[{ebv:.3f}, {ebvm:.3f}, {ebvp:.3f}]\"'
    

def initial():
    photometry_script('script_get_all_photometry.sh')
    setup_script('script_setup_all_fits.sh')
    run_script('script_run_all_fits.sh')
    print("run with:\n sh script_get_all_photometry.sh\n sh script_setup_all_fits.sh\n sh script_run_all_fits.sh\n\nInitial setup done!")

def photometry_script(fn):
    with open(fn, 'w') as f:
        # LMC targets
        for i in range(1, 440):
            f.write(f'speedyfit photometry "OGLE LMC-HB-{i:04d}"\n')
    
        # SMC targets
        for i in range(1, 41):
            f.write(f'speedyfit photometry "OGLE SMC-HB-{i:04d}"\n')

    

def setup_script(fn):
    extTable = Table.read('ogle_ext.txt',format='ascii')
    with open(fn,'w') as f:
        # write the fit script
        for i in range(1,440): 
            obj = f"OGLE LMC-HB-{i:04d}"
            myext = ext_str(obj,extTable,EVI_EBV_factor)
            f.write(f"speedyfit setup \"{obj}\" -grid kurucz_m05_31  --hb --nsample 10000 --nrelax 3000 --nwalkers 300 --location lmc --extinction {myext}\n")
            
            
        for i in range(1,41):
            obj = f"OGLE SMC-HB-{i:04d}"
            myext = ext_str(obj,extTable,EVI_EBV_factor)
            f.write(f"speedyfit setup \"{obj}\" -grid kurucz_m05_31  --hb --nsample 10000 --nrelax 3000 --nwalkers 300 --location smc --extinction {myext}\n")


def run_script(fn):
    with open(fn, 'w') as f:
        # LMC targets
        for i in range(1, 440):
            f.write(f'speedyfit fit "OGLE LMC-HB-{i:04d}_setup_kurucz_m05_31.yaml" --noplot\n')
    
        # SMC targets
        for i in range(1, 41):
            f.write(f'speedyfit fit "OGLE SMC-HB-{i:04d}_setup_kurucz_m05_31.yaml" --noplot\n')


def remove_duplicate_entries_by_distance(input_file, output_file):
    # Read the table
    tbl = Table.read(input_file, format='ascii.fixed_width', delimiter='|')
    print(tbl)

    # Remove rows with NaN in meas or emeas
    mask = ~np.isnan(tbl['meas']) & ~np.isnan(tbl['emeas'])
    tbl = tbl[mask]
    print(tbl)

    # Container for best rows
    rows_to_keep = []

    # Iterate over unique bands
    for band in np.unique(tbl['band']):
        group = tbl[tbl['band'] == band]
        if len(group) == 0:
            continue
        best_idx = np.argmin(group['distance'])
        rows_to_keep.append(group[best_idx])

    # Create cleaned table
    cleaned_tbl = vstack(rows_to_keep)
    print(cleaned_tbl)

    # Write to file
    cleaned_tbl.write(output_file, format='ascii.fixed_width', overwrite=True,delimiter='|')

    
def clean_photometry(sigma_threshold=3):
    filelist = sorted(glob.glob('*.phot'))
    for i,fn in enumerate(filelist):
        remove_duplicate_entries_by_distance(fn, fn+'clean')

def get_OCT(summaryfile):
    f = h5py.File(summaryfile, 'r')
    OCT = Table(f['O-C/Obs'][:])
    f.close()
    return OCT


def plot_resid(OCT):
    plt.errorbar(OCT['wave'],OCT['o-c'],OCT['o-c_err'],marker='o',ls='')
    plt.semilogx()
    plt.xlabel('wavelength [A]')
    plt.ylabel('residual [mag]')
    plt.gca().invert_yaxis()
    plt.axhline(0,ls='--',color='grey',zorder=0)

def ir_slope(summaryfile):
    """ returns ir slope, intercept in units of magnitudes / per dex wavelength """
    OCT = get_OCT(summaryfile)
    #plot_resid(OCT)
    sel_ir = (OCT['wave']>1e4) & (OCT['wave']<5e4)
    if(len(OCT[sel_ir])>0):
        my_x = np.log10(OCT['wave'][sel_ir]/1e4)
        my_y = OCT['o-c'][sel_ir]
        slope_ir, int_ir = np.polyfit(my_x,my_y , 1,w=1/OCT['o-c_err'][sel_ir])
    else:
        print("NO IR data for", summaryfile)
        slope_ir = 0
        int_ir = 0
        
    #xp = np.linspace(0,0.7)
    #plt.plot(1e4*10**xp,slope_ir*xp + int_ir)
    return slope_ir,int_ir

def get_ac(summaryfile):
    OCT = get_OCT(summaryfile)
    return weighted_lag1_autocorrelation( OCT['o-c'], OCT['o-c_err']) 

def weighted_lag1_autocorrelation(residuals, errors):
    """
    Compute the weighted lag-1 autocorrelation of residuals, accounting for measurement errors.

    Parameters:
    - residuals (array): Residual values.
    - errors (array): 1-sigma uncertainties associated with residuals.

    Returns:
    - float: Weighted lag-1 autocorrelation coefficient.
    """
    residuals = np.asarray(residuals)
    errors = np.asarray(errors)

    # Ensure valid length
    if len(residuals) < 2:
        return np.nan

    # Weights: inverse-variance
    w = 1.0 / (errors**2)

    # For lag-1, compare residuals[:-1] and residuals[1:]
    r1 = residuals[:-1]
    r2 = residuals[1:]
    w1 = w[:-1]
    w2 = w[1:]
    w_pair = np.sqrt(w1 * w2)  # Geometric mean weight for the pair

    # Weighted means
    mean1 = np.average(r1, weights=w1)
    mean2 = np.average(r2, weights=w2)

    # Weighted covariance
    cov = np.average((r1 - mean1) * (r2 - mean2), weights=w_pair)

    # Weighted variances
    var1 = np.average((r1 - mean1)**2, weights=w1)
    var2 = np.average((r2 - mean2)**2, weights=w2)

    if var1 == 0 or var2 == 0:
        return np.nan

    return cov / np.sqrt(var1 * var2)
            
def check():
    filelist = sorted(glob.glob('*.h5'))
    refitlist = []
    extTable = Table.read('ogle_ext.txt',format='ascii')
    results = []
    for i,fn in enumerate(filelist):
        obj = fn[0:16]
        slope,inter = ir_slope(fn)
        ac = get_ac(fn)
        results.append([obj,slope,ac])
        if (slope<-0.5) | (ac>0.5):
            print("for ",fn,"IR slope [mag/dex] = ",slope," ac = ",ac," REFITTING")
            refitlist.append(obj)
        else:
            print("for ",fn,"IR slope [mag/dex] = ",slope," ac = ",ac)


    rt = Table(np.array(results), names=['ID','irslope','ac'])
    rt.write('speedyfit_residual_check.csv',overwrite=True)
            
    with open('script_setup_all_refits.sh','w') as f:
        # write the refit script
        for obj in refitlist:
            loc = obj[5:8]
            if loc == 'LMC':
                location = 'lmc'
            if loc == 'SMC':
                location = 'smc'
            myext = ext_str(obj,extTable,EVI_EBV_factor)
            cmd = f"speedyfit setup \"{obj}\" -grid kurucz_m05_31  --hb --nsample 10000 --nrelax 3000 --nwalkers 300 --location {location} --exclude_ir --extinction {myext}\n"
            f.write(cmd)

    
    with open('script_run_all_refits.sh','w') as f:
        # write the refit script
        for obj in refitlist:
            loc = obj[5:8]
            if loc == 'LMC':
                location = 'lmc'
            if loc == 'SMC':
                location = 'smc'

            cmd = f"speedyfit fit \"{obj}_setup_kurucz_m05_31.yaml\" --noplot\n" 
            f.write(cmd)

    print("run with:\n sh script_setup_all_refits.sh\n sh script_run_all_refits.sh\n\nCheck done!")


def table():
    files = sorted(glob.glob("OGLE*results_kurucz_m05_31.csv"))
    print("Grouping files (len): ", len(files))
    tables = []

    for fname in files:
        tab = Table.read(fname, format='csv')

        # Extract ID from the filename (first 16 characters)
        object_id = fname[:16]

        # Add new column with the same ID repeated for all rows in the table
        tab['ID'] = [object_id] * len(tab)

        tables.append(tab)

    # Stack all tables
    combined_table = vstack(tables)

    print(combined_table)
    combined_table.write("hb_sed_speedyfit.csv", format="csv", overwrite=True)




def main():
    parser = argparse.ArgumentParser(description='Generate speedyfit scripts')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcommand: initial
    subparsers.add_parser('initial', help='Generate phot,setup,run scripts')
    
    # Subcommand: cleanphot
    subparsers.add_parser('cleanphot', help='check for duplicates in photometry')
   
    # Subcommand: check
    subparsers.add_parser('check',help='Run IR slope residual check, generate rerun setup, fit scripts')

    # Subcommand: table
    subparsers.add_parser('table', help='Generate final results table by grouping data')

    args = parser.parse_args()

    if args.command == 'initial':
        initial()
    elif args.command =='cleanphot':
        clean_photometry()
    elif args.command == 'check':
        check()
    elif args.command =='table':
        table()
    else:
        print("ERROR not recognized option")

if __name__ == '__main__':
    main()
