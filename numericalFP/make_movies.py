#!/opt/local/bin/python
import os

fig_type = 'png'
vcodec = 'wmv2'
fps = 24

plotDir = '../results/plot/'
dstDir = plotDir + 'numericalFP/'

#q = 0.5
q = 1.
#q = 2

#nx0 = 100
nx0 = 200

prop = "numFP"
#prop = "statDist"

dst_movie = '%s%s_hopf_nx%d_q%02d.mpg' % (dstDir, prop, nx0, int(q * 10))
files = '%s_hopf_nx%d_k*_mu*_q%02d.%s' % (prop, nx0, int(q * 10), fig_type)
smf = '"mf://' + dstDir + files + '"'
sexe = 'mencoder ' + smf + ' -mf type=' + fig_type + ':' + 'fps=' + str(fps) + ' -ovc lavc -lavcopts vcodec=' + vcodec + ' -oac copy -o ' + dst_movie
os.system(sexe)

