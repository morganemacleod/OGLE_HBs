python generate_hb_scripts.py initial
#sh script_get_all_photometry.sh
#python generate_hb_scripts.py cleanphot

#sh script_setup_all_fits.sh
split -l 100 -a 1 script_setup_all_fits.sh tmp_
ls tmp_* | parallel sh {}

#sh script_run_all_fits.sh
split -l 100 -a 1 script_run_all_fits.sh tmp_
ls tmp_* | parallel sh {}

python generate_hb_scripts.py check
sh script_setup_all_refits.sh
#sh script_run_all_refits.sh
split -l 25 -a 1 script_run_all_refits.sh tmp_
ls tmp_* | parallel sh {}

python generate_hb_scripts.py table
