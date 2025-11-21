import numpy as np
import os
import glob
from plot_imports import *
import matplotlib.pyplot as plt
import subprocess
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from allosteric_utils import Allosteric


def read_log(filename):
    """
    Reads LAMMPS log file and returns data between the 'Step' and 'Loop' lines.

    Parameters
    ----------
    filename : str
        The name of the log file to read.

    Returns
    -------
    data : ndarray
        The parsed log data from 'Step' to 'Loop'.
    cols : list of str
        Column names.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = []
    cols = []
    reading = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Step"):
            cols = line.split()
            reading = True
            continue
        if line.startswith("Loop"):
            break
        if reading:
            data_lines.append([float(x) for x in line.split()])

    if not data_lines:
        raise ValueError("No data found between 'Step' and 'Loop' in log file.")

    data = np.array(data_lines)
    return data, cols

def read_data(filename, graph):
	'''Read a LAMMPS data file and update a graph based on its contents.

	The bond stiffnesses and rest lengths are set based on the datafile specifications.
	   
	Parameters
	----------
	filename : str
		The name of the file to read.
	graph : networkx.graph
		The graph to update.
	'''

	with open(filename) as f:
		line = f.readline()
		while len(line.strip().split()) < 1 or line.strip().split()[0] != 'Bond':
			line = f.readline()
		f.readline() # empty space
		for i, edge in enumerate(graph.edges(data=True)):
			line = f.readline()
			idx, hk, l = np.array(line.strip().split())[:3].astype(float)
			k = 2*hk
			edge[2]['stiffness'] = k
			edge[2]['length'] = l

def read_data_new(filename, allo,train,Adam=False):
	'''Read a LAMMPS data file and update a graph based on its contents.

	The bond stiffnesses and rest lengths are set based on the datafile specifications.
	   
	Parameters
	----------
	filename : str
		The name of the file to read.
	graph : networkx.graph
		The graph to update.
	'''

	with open(filename) as f:
		line = f.readline()
		while len(line.strip().split()) < 1 or line.strip().split()[0] != 'Bond':
			line = f.readline()
			
		f.readline() # empty space
		for i, edge in enumerate(allo.graph.edges(data=True)):
			line = f.readline()
			idx, hk, l = np.array(line.strip().split())[:3].astype(float)
			if Adam:
				m,v= np.array(line.strip().split())[11:13].astype(float)


			k = 2*hk
			edge[2]['stiffness'] = k
			edge[2]['length'] = l
			if Adam:
				edge[2]['m']=m
				edge[2]['v']=v
		while len(line.strip().split()) < 1 or line.strip().split()[0] != 'Atoms':
			line = f.readline()
		f.readline() # empty space

		for i in range(allo.n): 
			if train:
				line = f.readline()
				id,_,_, xc, yc, zc,_,_,_ = np.array(line.strip().split())[:9].astype(float)
				allo.pts_c[i,0] = xc
				allo.pts_c[i,1] = yc
				allo.pts_c[i,2] = zc
			line = f.readline()
			id,_,_, x, y, z,_,_,_ = np.array(line.strip().split())[:9].astype(float)
			allo.pts[i,0] = x
			allo.pts[i,1] = y
			allo.pts[i,2] = z
			

def read_dim(filename):
	'''Read a LAMMPS input file to parse out the dimension.

	Parameters
	----------
	filename : str
		The name of the file to read.

	Returns
	-------
	int
		The dimension of the system.
	'''
	with open(filename) as f:
		line = f.readline()
		while len(line.strip().split()) < 1 or line.strip().split()[0] != 'dimension':
			line = f.readline()
	return int(line.strip().split()[1])


def setup_run_new(allo, odir, prefix, lmp_path, duration, frames, applied_args, train=0, method=None, eta=1., alpha=1e-3, vmin=1e-3, temp=0, dt=0.005, hours=24,seed=12,beta1=0.9, beta2=0.999,WCA=False,DOC=True,Twin=False,phase=1):
	'''Set up a complete LAMMPS simulation in a directory.
	   
	Parameters
	----------
	allo : Allosteric
		The Allosteric object to simulate.
	odir : str
		The path to the directory.
	prefix : str
		The file prefix to use for data, input, dump, and logfiles.
	lmp_path : str
		The path to the LAMMPS executable.
	duration : float
		The final integration time.
	frames : int
		The number of output frames to produce (excluding initial frame).
	applied_args : tuple
		Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
	train : int, optional
		Training mode. 0 = no training, 1 = l-model, 2 = k-model.
	method : str, optional
		Training method to use. Options are 'aging' or 'learning'.
	eta : float, optional
		The learning rate by which the clamped state target strain approaches the final desired strain.
	alpha : float, optional
		The aging rate.
	vmin : float, optional
		The smallest allowed value for each learning degree of freedom.
	temp : float, optional
		The temperature setting, in LJ units. If zero (default), an athermal simulation is performed.
	dt : float, optional
		Integration step size.
	hours : int, optional
		The number of hours to allocate for the job.
	'''
	

	datafile = prefix+'.data'
	infile = prefix+'.in'
	logfile = prefix+'.log'
	jobfile = 'job.sh'

	if odir[-1] != '/' : odir += '/'
	if not os.path.exists(odir):
		os.makedirs(odir)

	if train:
		allo.write_lammps_data_learning(odir+datafile, 'Allosteric network', applied_args,
										train=train, method=method, eta=eta, alpha=alpha, vmin=vmin,
										beta1=beta1, beta2=beta2, dt=dt,WCA=WCA,phase=phase)
	else:
		allo.write_lammps_data(odir+datafile, 'Allosteric network', applied_args)

	
	allo.write_lammps_input_new(odir+infile, datafile, duration, frames, temp=temp, method=method,dt=dt,seed=seed,WCA=WCA,DOC=DOC,Twin=Twin)
	allo.save(odir+'allo.txt') # do this last, because it resets init!!

	cmd = lmp_path+' -i '+infile+' -log '+logfile

	allo.write_job(odir+jobfile, prefix+'_test', hours, cmd)
	# submit job together
	with open('tasks.sh', 'a') as f:
		f.write(f"cd {odir}\n")
		f.write(f"sbatch ./{jobfile}\n")
	print("LAMMPS simulation with Bond Info set up in directory: {:s}".format(odir))


def setup_test(allo, odir, prefix, lmp_path, duration,applied_args,lines=10000, temp=0, dt=0.005, hours=24,seed=12,WCA=True):
	datafile = prefix+'.data'
	infile = prefix+'.in'
	logfile = prefix+'.log'
	jobfile = 'job.sh'

	if odir[-1] != '/' : odir += '/'
	if not os.path.exists(odir):
		os.makedirs(odir)
	
	allo.write_lammps_data(odir+datafile, 'Allosteric network', applied_args) 
	allo.write_lammps_input_test(odir+infile, datafile, duration, lines=lines,temp=temp, dt=dt, seed=seed, WCA=WCA)
	allo.save(odir+'allo.txt') # do this last, because it resets init!!

	cmd = lmp_path+' -i '+infile+' -log '+logfile

	allo.write_job(odir+jobfile, prefix+'_test', hours, cmd)
	print("LAMMPS simulation set up in directory: {:s}".format(odir))


def load_frame(odir, frame=200,train=True,Adam=False):
	if odir[-1] != '/' : odir += '/'
	netfile = glob.glob(odir+'*.txt')[0]
	datafile = glob.glob(odir+f'step{frame:d}.bond')[0]
	infile = glob.glob(odir+'*.in')[0]
	
	
	allo = Allosteric(netfile)
	dim = read_dim(infile)
	if dim != allo.dim:
		raise ValueError("Dimension mismatch between LAMMPS simulation (d={:d}) and network file (d={:d}).".format(dim,allo.dim))
	read_data_new(datafile, allo,train,Adam=Adam)
	return allo


def get_traj(odir,nframes=200,train=True,Adam=False):
	for i in range(1,nframes+1):
		allo = load_frame(odir, frame=i,train=train,Adam=Adam)
		stiffness = np.array([edge[2]['stiffness'] for edge in allo.graph.edges(data=True)])
		if Adam:
			m = np.array([edge[2]['m'] for edge in allo.graph.edges(data=True)])
			v = np.array([edge[2]['v'] for edge in allo.graph.edges(data=True)])
		
		if i == 1:
			r_traj = np.copy(allo.pts)
			if train:
				rc_traj = np.copy(allo.pts_c)
				k_traj = np.copy(stiffness)
				if Adam:
					m_traj = np.copy(m)
					v_traj = np.copy(v)
				else:
					m_traj = None
					v_traj = None
			else:
				rc_traj = None
				k_traj = None
		else:
			r_traj = np.concatenate((r_traj, allo.pts), axis=0)
			if train:
				rc_traj = np.concatenate((rc_traj, allo.pts_c), axis=0)
				k_traj = np.concatenate((k_traj, stiffness), axis=0)
				if Adam:
					m_traj = np.concatenate((m_traj, m), axis=0)
					v_traj = np.concatenate((v_traj, v), axis=0)
				
	if not train:
		return r_traj
	elif not Adam:
		return r_traj, rc_traj, k_traj
	else:
		return r_traj, rc_traj, k_traj, m_traj, v_traj

def get_clusters(data, n, seed=12):
	'''Get k-means clusters.

	Parameters
	----------
	data : ndarray
		The data to cluster.
	n : int
		The number of clusters.
	seed : int, optional
		The random seed.

	Returns
	-------
	ndarray
		The cluster id to which each data point belongs.
	float
		The silhouette score of the clustering.

	'''
	km = KMeans(n_clusters=n, random_state=seed)
	labels = km.fit_predict(data.reshape(-1,1))
	score = silhouette_score(data.reshape(-1,1), labels)
	return labels, score

def setup_quench(allo,odir,lmp_path,applied_args,temp,etol=0,ftol=1e-10,maxiter=50000,dt=0.005):
	'''Set up a quench simulation in a directory.

	This function creates a directory for the quench simulation, sets up the
	LAMMPS input file, and writes the necessary files for the simulation.
	'''
	

	odir = odir+'/quench/'
	if not os.path.exists(odir):
		os.makedirs(odir)
	prefix = 'quench'

	datafile = prefix+'.data'
	infile= prefix+'.in'
	dumpfile = prefix+'.dump'
	logfile = prefix+'.log'
	jobfile = 'job.sh'
	
	
	allo.write_lammps_data(odir+datafile, 'Allosteric network', applied_args)
	allo.write_quench_input(odir+infile, datafile, dumpfile,temp, etol=etol, ftol=ftol, maxiter=maxiter, dt=dt)
	
	allo.save(odir+'allo.txt') # do this last, because it resets init!!

	cmd = lmp_path+' -i '+infile+' -log '+logfile

	allo.write_job(odir+jobfile, prefix+'_test', 1, cmd)#hours=1, quench should be fast
	
	with open('tasks.sh', 'a') as f:
		f.write(f"cd {odir}\n")
		f.write(f"sbatch ./{jobfile}\n")
	
	print("LAMMPS quench simulation set up in directory: {:s}".format(odir))

def submit_job_array(odir_list, lmp_path,prefix, array_name="job_array", max_concurrent=50, logfile="tasks.sh"):
	"""
	Generate and submit a Slurm job array script.

	Parameters
	----------
	odir_list : list[str]
		List of directories where the jobs should be run.
	jobfile : str
		Name of the job script (e.g., 'run.slurm' or 'job.sh') located in each directory.
	array_name : str, optional
		Name prefix for the job array (default: 'job_array').
	max_concurrent : int, optional
		Maximum number of simultaneous jobs in the array (default: 10).
	logfile : str, optional
		File name for the generated submission script (default: 'tasks.sh').
	"""

	missing = [d for d in odir_list if not os.path.isdir(d)]
	if missing:
		print("The following directories do NOT exist:")
		for d in missing:
			print(f"   - {d}")
		print("Aborting submission.")
		return
	else:
		print(f"All {len(odir_list)} directories exist.")

	n_jobs = len(odir_list)
	array_script = f"{array_name}.slurm"
	infile = prefix+'.in'
	logfile = prefix+'.log'
	cmd = lmp_path+' -i '+infile+' -log '+logfile
	# Write the array job script
	with open(array_script, "w") as f:
		f.write("#!/bin/bash\n")
		f.write("#SBATCH --partition=liu_compute \n")
		f.write(f"#SBATCH --job-name={array_name}\n")
		f.write(f"#SBATCH --array=0-{n_jobs-1}%{max_concurrent}\n")
		f.write("#SBATCH --output=~/messages/slurm-%A_%a.out\n\n")

		f.write("ODIRS=(" + " ".join(odir_list) + ")\n")
		f.write("cd ${ODIRS[$SLURM_ARRAY_TASK_ID]}\n")
		f.write(f"{cmd}\n")

	# Log submission commands
	with open(logfile, "a") as f:
		f.write(f"sbatch {array_script}\n")


	print(f"created job array with {n_jobs} jobs, max running {max_concurrent}. To submit, run: sbatch {array_script} ")

