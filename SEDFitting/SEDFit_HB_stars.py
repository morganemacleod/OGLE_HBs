import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline 
from SEDFit.sed import SEDFit
from astropy.table import Table


def read_ogle(filename):
    # NOTE: need to apply diff dist moduli 
    def get_loc(name):
        return name[5:8]
    get_loc=np.vectorize(get_loc)


    hb = Table.read(filename,header_start=6,data_start=7,format='ascii')
    hb['loc'] = get_loc(hb['ID'])
    return hb

def print_info(x):
    
    print(" ------ fit info: ------")
    print("PARS:",x.fit_par, x.fit_par_err)
    print("Distance: {} pc".format(x.getdist()))
    print("AV: {}+/-{} mag".format( np.round(x.getav(),3), np.round(x.fit_par_err[0],3) ))
    print("Radius: {}+/-{} Rsun".format( np.round(x.getr()[0],2), np.round(x.fit_par_err[1],2) ))
    print("Teff: {}+/-{} K".format(np.round(x.getteff()[0]), np.round(x.fit_par_err[2]) ))
    print("Log g: {}+/-{} ".format(np.round(x.getlogg()[0],2),np.round(x.fit_par_err[3],2) ))
    print("Fe/H: {}".format(x.getfeh()))
    print("Chi squared/N: {}".format( np.round(x.chi2oN,3) )  )
    print("IR slope = ",np.round(ir_slope(x),4),"dex/micron")
    print(" ------------")

def ir_slope(x):
    # check for structured residual
    sel_ir = (x.sed['la']>1e4) & (x.sed['la']<5e4)
    my_x = x.sed['la'][sel_ir]/1.e4
    my_y = (x.sed['flux']-x.sed['model'])[sel_ir]
    slope_ir, int_ir = np.polyfit(my_x,my_y , 1)
    return slope_ir

def do_fit(x,use_idx):
    pars = x.fit( use_mag = use_idx , fitfeh=False,fitdist=False) 
    x.chi2oN = x.getchisq(idx=use_idx)
    x.sed['model']=x.mags
    x.fit_par = pars[0]
    x.fit_par_err = np.sqrt(np.diag(pars[1]))
    #print("PARS:",fit_par,fit_par_err )
    print_info(x)
    return 
       
    
def do_fit_reject_outliers(x,AA_cut_IR,chisqr_threshold):
    use_idx = np.where(x.sed['la']<AA_cut_IR)[0]
    do_fit(x,use_idx)
    # check for outliers
    zscore    =  np.abs(x.sed['flux']-x.sed['model'])/x.sed['eflux']
    # goodness of fit check for iterative removal of indicies... 
    if((np.max(zscore) > 5) & (x.chi2oN >chisqr_threshold)):
           for j in range(5):
                print("... doing outlier removal iteration:",j)
                zscore    =  np.abs(x.sed['flux']-x.sed['model'])/x.sed['eflux']   
                if np.max(zscore)<3:
                    break
                else:
                    print("outliers detected, max zscore=",np.max(zscore))
                    zscore_thresh = np.sort(zscore)[::-1][j]
                    zscore_thresh = np.where(zscore_thresh<5,5,zscore_thresh) # don't cut if < 5 sigma away... 
                    use_idx   = np.where( (x.sed['la']<AA_cut_IR) & (zscore<zscore_thresh) )[0]
                    # refit 
                    do_fit(x,use_idx)
                    if(x.chi2oN < chisqr_threshold):
                        break
    return use_idx
    

def do_lmcsmc_fit(ra,dec,
                  gal='LMC',AA_cut_IR=1e99,ID=None,
                 minav=0,maxav=3.1,chisqr_threshold=1e99,
                 download_flux=False,radius=1):
    if gal=='LMC':
        print("Fitting for LMC location")
        mydist = 49.97e3 #pc
        myfeh  = -0.7 #-0.42 #dex #Fig 11, https://arxiv.org/abs/2006.02448 @ 4deg 
    elif gal=='SMC':
        print("Fitting for SMC location")
        mydist = 62.44e3 #pc
        myfeh = -1.05 #-0.94 #Fig 11, https://arxiv.org/abs/2006.02448 @ 4deg 
    else:
        print("galaxy not recognized (LMC,SMC) are options, entered:",gal)
        return
    
    print(' at these coords, the min Av =',np.round(minav,2)," and the max Av =",np.round(maxav,2))
    
    crd=(str(ra)+'_'+str(dec)+'_'+str(radius)).replace(':','_')
    if ID is None:
        flux_filename=crd+'_sed.csv'
        plot_filename=crd+'_sedplot.pdf'
        vizier_filename=crd+'_vizier_sed.vot'
    else:
        flux_filename=ID+'_sed.csv'
        plot_filename=ID+'_sedplot.pdf'
        vizier_filename=ID+'_vizier_sed.vot'
    print("Saving output to:",flux_filename,plot_filename,vizier_filename)
    
    # setup, add guesses 
    x=SEDFit(ra,dec,radius,
             use_gaia_params=False,
             download_flux=download_flux,
             tmass=True,cousins=True,
             gaia=True,galex=False,
             johnson=True,panstarrs=True,
             sdss=True,wise=True,
             xmm=True,spitzer=True,
             flux_filename=flux_filename,
             vizier_filename=vizier_filename,
             setmaxav=maxav,
             min_eflux = 0.04,
             fit_dist=False)
    
    print(x.sed)
    # initial fit
    x.addguesses(r=[20],teff=[6000],logg=0.5,dist=mydist,feh=myfeh,av=(minav+maxav)/2)
    x.addrange(logg=[0,3],av=[minav,maxav],teff=[3000,35000] ) #,dist=[mydist-1e-3,mydist+1e-3]
    use_idx = do_fit_reject_outliers(x,AA_cut_IR,chisqr_threshold)
    
    # Log g consistency check
    Lest = 4*np.pi*(x.getr()[0]*7e10)**2 * 5.67051e-05 * x.getteff()[0]**4 / 3.846e33
    Mest = Lest**(1/3.5)
    loggfit = x.getlogg()[0]
    print(" estimated mass (msun) =",np.round(Mest,1) ) 
    loggmax = float(np.round(np.log10( 6.67e-8*2*Mest*2e33/(x.getr()[0]*7e10)**2 ),4))
    if (loggfit > loggmax):
        print("setting logg_max =",loggmax)
        x.addguesses(logg=loggmax/2)
        x.addrange(logg=[0,loggmax],dist=[mydist-1e-3,mydist+1e-3],av=[minav,maxav] )
      
    # Hot star
    if ((x.getteff()[0]>6250.0) & ((np.abs(ir_slope(x)) > 0.1/4.) | (loggfit > loggmax) ) ):
        print("Hot star, IR excess detected, restricting fitting range to optical")
        use_idx = do_fit_reject_outliers(x,1e4,chisqr_threshold)
    
    # cool star 
    if ((x.getteff()[0]<6250.0) & (loggfit > loggmax) ):
        print("Cool star, refitting for loggmax")
        use_idx = do_fit_reject_outliers(x,AA_cut_IR,chisqr_threshold)
    
    # make a plot
    x.makeplot(file=plot_filename,idx=use_idx,title=ID)
    
    
    return x 


mode = 'redo' # 'initial' or 'redo'
my_radius_default = 2

special_radii = {    'OGLE-LMC-HB-0140':1,
                     'OGLE-LMC-HB-0309':1,
                     'OGLE-LMC-HB-0310':1,
                     'OGLE-LMC-HB-0392':1,
                     'OGLE-SMC-HB-0003':0.5,
                     'OGLE-SMC-HB-0005':0.5,
                     'OGLE-SMC-HB-0006':1}
redo_list = []
    

if mode=='initial':
    # Read and select LMC, SMC
    hb = read_ogle("query_1683478037.14515r.txt")
    hb = hb[hb['loc']!='BLG'].copy()

    # read extinction https://ogle.astrouw.edu.pl/cgi-ogle/get_ms_ext.py
    ext = Table.read('ogle_ext.txt',format='ascii')

    if len(ext)!=len(hb):
        raise Exception("TABLES DO NOT SEEM TO MATCH")

    #initialize new columns
    hb['sed_dist'] = 0.0
    hb['sed_av'] = 0.0
    hb['sed_av_err'] = 0.0
    hb['sed_radius']=0.0
    hb['sed_radius_err']=0.0
    hb['sed_teff'] = 0.0
    hb['sed_teff_err'] = 0.0
    hb['sed_logg'] = 0.0
    hb['sed_logg_err'] = 0.0
    hb['sed_feh'] = 0.0
    hb['sed_irslope']=0.0
    hb['sed_chi2oN'] = 0.0
    hb['Av_median'] = 2.25 * ext['E(V-I)']
    hb['Av_minus_two_sigma'] =2.25 * ( ext['E(V-I)'] - 2*ext['-sigma1'] )
    hb['Av_plus_two_sigma'] = 2.25 * ( ext['E(V-I)'] + 2*ext['+sigma2'])
    hb['Av_minus_two_sigma'] = np.where(hb['Av_minus_two_sigma']<0, 0, hb['Av_minus_two_sigma'])
elif mode=='redo':
    hb = Table.read('ogle_hb_sedfit_r2.csv')
    redo_list = [ 'OGLE-LMC-HB-0140','OGLE-LMC-HB-0309','OGLE-LMC-HB-0310',
                  'OGLE-LMC-HB-0392','OGLE-SMC-HB-0003','OGLE-SMC-HB-0005','OGLE-SMC-HB-0006']

else:
    print("MODE NOT RECOGNIZED")
    

 
    
failed = []
# do fitting
for i in range(len(hb)):
    print("========= i=",i,"============================================")
    if(mode=='redo') & (hb['ID'][i] not in redo_list):
        continue
    else:
        try:
            if hb['ID'][i] in special_radii:
                my_radius = special_radii[hb['ID'][i]]
            else:
                my_radius = my_radius_default
            
            x=do_lmcsmc_fit(hb['RA'][i],hb['Decl'][i],gal=hb['loc'][i],ID=hb['ID'][i],
                                    AA_cut_IR=4e4,
                                    minav=hb['Av_minus_two_sigma'][i],maxav=hb['Av_plus_two_sigma'][i],
                                    chisqr_threshold=7.0,
                                    download_flux=True,
                                    radius=my_radius)
    
            hb['sed_dist'][i] = x.getdist()
            hb['sed_av'][i] = x.getav()
            hb['sed_radius'][i] = x.getr()[0]
            hb['sed_teff'][i] = x.getteff()[0]
            hb['sed_logg'][i] = x.getlogg()[0]
            hb['sed_feh'][i] = x.getfeh()
            hb['sed_chi2oN'][i] = x.chi2oN
            hb['sed_irslope'][i] = ir_slope(x)
            hb['sed_av_err'][i] = x.fit_par_err[0]
            hb['sed_radius_err'][i] = x.fit_par_err[1]
            hb['sed_teff_err'][i] = x.fit_par_err[2]
            hb['sed_logg_err'][i] = x.fit_par_err[3]
        except:
            print("INDEX:",i, "FAILED,added to list")
            failed.append(i)
        
hb['sed_L'] = 4*np.pi*(hb['sed_radius']*7e10)**2 * 5.67051e-05 * hb['sed_teff']**4 / 3.846e33
    
# write the output
print("FAILED INDICIES: ",failed)
hb.write('ogle_hb_sedfit_r2r.csv',overwrite=True)
