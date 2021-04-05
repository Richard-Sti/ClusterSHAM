from PySHAM import surveys

survey = surveys.NSASurvey('ELPETRO')
nthreads = 12
outfolder = '/mnt/zfsusers/rstiskalek/pysham/results/NSAmatch_ELPETRO/'

num_density = surveys.NumberDensity(survey, outfolder, nthreads)

__ = num_density.number_density()
