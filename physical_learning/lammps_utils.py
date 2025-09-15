import numpy as np
import os
import glob
from plot_imports import *
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from allosteric_utils import Allosteric

def read_dump(filename):
	'''Read a LAMMPS dumpfile.
	   
	Parameters
	----------
	filename : str
		The name of the file to read.

	Returns
	-------
	ndarray
		The (x,y) coordinates for each of n points over all output frames.
	'''

	with open(filename, 'r') as f:
		lines = f.readlines()
		if len(lines[0].split()) > 1:
			n = int(lines[3].split()[0])
			h = 9
			m = n+9
		else:
			n = int(lines[0].split()[0])
			h = 2
		m = n+h
		frames = len(lines)//m

	traj = np.zeros((frames,n,3))
	vtraj = np.zeros((frames,n,3))
	vx, vy, vz = 0, 0, 0
	for fr in range(frames):
		for i in range(n):
			line = lines[m*fr+h+i]
			line = np.array(line.strip().split()).astype(float)
			if len(line) > 4:
				x, y, z, vx, vy, vz = line[-6], line[-5], line[-4], line[-3], line[-2], line[-1]
			else:
				x, y, z = line[-3], line[-2], line[-1]
			traj[fr,i,0] = x
			traj[fr,i,1] = y
			traj[fr,i,2] = z
			vtraj[fr,i,0] = vx
			vtraj[fr,i,1] = vy
			vtraj[fr,i,2] = vz

	return traj, vtraj


def read_dump_bondinfo(filename):
	'''Read a LAMMPS dumpfile.
	   
	Parameters
	----------
	filename : str
		The name of the file to read.

	Returns
	-------
	ndarray
		The (x,y) coordinates for each of n points over all output frames.
	'''

	with open(filename, 'r') as f:
		lines = f.readlines()
		if len(lines[0].split()) > 1:
			n = int(lines[3].split()[0])
			h = 9
			m = n+9
		else:
			n = int(lines[0].split()[0])
			h = 2
		m = n+h
		frames = len(lines)//m

	dist = np.zeros((frames,n,1))
	engpot = np.zeros((frames,n,1))
	
	for fr in range(frames):
		for i in range(n):
			line = lines[m*fr+h+i]
			line = np.array(line.strip().split()).astype(float)
			id, c1, c2 = line[-3], line[-2], line[-1]
			dist[fr,i,0] = c1
			engpot[fr,i,0] = c2
			
	return dist, engpot


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

def read_data_new(filename, allo,train):
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
			k = 2*hk
			edge[2]['stiffness'] = k
			edge[2]['length'] = l
		while len(line.strip().split()) < 1 or line.strip().split()[0] != 'Atoms':
			line = f.readline()
		f.readline() # empty space

		for i in range(allo.n): 
			line = f.readline()
			id,_,_, x, y, z,_,_,_ = np.array(line.strip().split())[:9].astype(float)
			allo.pts[i,0] = x
			allo.pts[i,1] = y
			allo.pts[i,2] = z
			if train:
				line = f.readline()
				id,_,_, xc, yc, zc,_,_,_ = np.array(line.strip().split())[:9].astype(float)
				allo.pts_c[i,0] = xc
				allo.pts_c[i,1] = yc
				allo.pts_c[i,2] = zc

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

def setup_run(allo, odir, prefix, lmp_path, duration, frames, applied_args, train=0, method=None, eta=1., alpha=1e-3, vmin=1e-3, temp=0, symmetric=False,doc_bond=False, dt=0.005, hours=24,seed=12):
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
	symmetric : bool, optional
		Whether to introduce a symmetric state for training with a different set of boundary conditions. Default is False.
	dt : float, optional
		Integration step size.
	hours : int, optional
		The number of hours to allocate for the job.
	'''
	

	datafile = prefix+'.data'
	infile = prefix+'.in'
	dumpfile = prefix+'.dump'
	logfile = prefix+'.log'
	jobfile = 'job.sh'

	if odir[-1] != '/' : odir += '/'
	if not os.path.exists(odir):
		os.makedirs(odir)

	if train:
		allo.write_lammps_data_learning(odir+datafile, 'Allosteric network', applied_args,
										train=train, method=method, eta=eta, alpha=alpha, vmin=vmin,
										symmetric=symmetric, dt=dt)
	else:
		allo.write_lammps_data(odir+datafile, 'Allosteric network', applied_args)
	allo.write_lammps_input(odir+infile, datafile, dumpfile, duration, frames, temp=temp, method=method, symmetric=symmetric,doc_bond=doc_bond, dt=dt,seed=seed)
	allo.save(odir+'allo.txt') # do this last, because it resets init!!

	cmd = lmp_path+' -i '+infile+' -log '+logfile

	allo.write_job(odir+jobfile, prefix+'_test', hours, cmd)
	# submit job together
	with open('tasks.sh', 'a') as f:
		f.write(f"cd {odir}\n")
		f.write(f"sbatch ./{jobfile}\n")
	
	print("LAMMPS simulation set up in directory: {:s}".format(odir))


def setup_run_new(allo, odir, prefix, lmp_path, duration, frames, applied_args, train=0, method=None, eta=1., alpha=1e-3, vmin=1e-3, temp=0, symmetric=False,dt=0.005, hours=24,seed=12,beta1=0.9, beta2=0.999,WCA=False,DOC=True):
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
	symmetric : bool, optional
		Whether to introduce a symmetric state for training with a different set of boundary conditions. Default is False.
	dt : float, optional
		Integration step size.
	hours : int, optional
		The number of hours to allocate for the job.
	'''
	

	datafile = prefix+'.data'
	infile = prefix+'.in'
	dumpfile = prefix+'.dump'
	logfile = prefix+'.log'
	jobfile = 'job.sh'

	if odir[-1] != '/' : odir += '/'
	if not os.path.exists(odir):
		os.makedirs(odir)

	if train:
		allo.write_lammps_data_learning(odir+datafile, 'Allosteric network', applied_args,
										train=train, method=method, eta=eta, alpha=alpha, vmin=vmin,
										symmetric=symmetric, beta1=beta1, beta2=beta2, dt=dt,WCA=WCA)
	else:
		allo.write_lammps_data(odir+datafile, 'Allosteric network', applied_args)
	allo.write_lammps_input_new(odir+infile, datafile, dumpfile, duration, frames, temp=temp, method=method, symmetric=symmetric, dt=dt,seed=seed,WCA=WCA,DOC=DOC)
	allo.save(odir+'allo.txt') # do this last, because it resets init!!

	cmd = lmp_path+' -i '+infile+' -log '+logfile

	allo.write_job(odir+jobfile, prefix+'_test', hours, cmd)
	# submit job together
	with open('tasks.sh', 'a') as f:
		f.write(f"cd {odir}\n")
		f.write(f"sbatch ./{jobfile}\n")
	print("LAMMPS simulation with Bond Info set up in directory: {:s}".format(odir))


def load_run(odir, history=True):
	'''Load a complete LAMMPS simulation from its directory.

	The directory should contain an Allosteric network file, LAMMPS datafile,
	dumpfile, and logfile.

	Parameters
	----------
	odir : str
		The path to the directory.
	just_allo : bool, optional
		If True, only load the Allosteric object without the simulation history.

	Returns
	-------
	allo : Allosteric
		Allosteric Class object with network set up according to provided LAMMPS datafile,
		with simulation history loaded from dumpfile (if present).
	'''

	# collect all filenames
	if odir[-1] != '/' : odir += '/'
	netfile = glob.glob(odir+'*.txt')[0]
	datafile = glob.glob(odir+'*.data')[0]
	infile = glob.glob(odir+'*.in')[0]
	logfile = glob.glob(odir+'*.log')[0]

	dumpfiles = glob.glob(os.path.join(odir, '*.dump'))

	# Separate bondinfo.dump and other dumps
	has_bondinfo = any(os.path.basename(f) == 'bondinfo.dump' for f in dumpfiles)
	other_dumps = [f for f in dumpfiles if os.path.basename(f) != 'bondinfo.dump']
	has_other_dump = len(other_dumps) > 0

	if has_other_dump:
		dumpfile = other_dumps[0]

	allo = Allosteric(netfile)
	dim = read_dim(infile)
	if dim != allo.dim:
		raise ValueError("Dimension mismatch between LAMMPS simulation (d={:d}) and network file (d={:d}).".format(dim,allo.dim))
	read_data(datafile, allo.graph)
	
	if not has_other_dump or history==False:
		return allo, None, None,None,None

	
	data, cols = read_log(logfile)
	if 'Time' in cols:
		allo.t_eval = data[:, cols.index('Time')]

	traj, vtraj = read_dump(dumpfile)
	if traj.shape[1] == allo.n: # free state only
		allo.traj = np.copy(traj)
		allo.vtraj = np.copy(vtraj)
		allo.traj_c = np.copy(traj)
		allo.vtraj_c = np.copy(vtraj)
		allo.traj_s = np.copy(traj)
		allo.vtraj_s = np.copy(vtraj)
		allo.traj_sc = np.copy(traj)
		allo.vtraj_sc = np.copy(vtraj)
	elif traj.shape[1] == 2*allo.n: # free and clamped states
		allo.traj = np.copy(traj[:,1::2,:])
		allo.vtraj = np.copy(vtraj[:,1::2,:])
		allo.traj_c = np.copy(traj[:,::2,:])
		allo.vtraj_c = np.copy(vtraj[:,::2,:])
		allo.traj_s = np.copy(traj[:,1::2,:])
		allo.vtraj_s = np.copy(vtraj[:,1::2,:])
		allo.traj_sc = np.copy(traj[:,::2,:])
		allo.vtraj_sc = np.copy(vtraj[:,::2,:])
	else: # symmetric free and clamped states
		allo.traj = np.copy(traj[:,1::4,:])
		allo.vtraj = np.copy(vtraj[:,1::4,:])
		allo.traj_c = np.copy(traj[:,::4,:])
		allo.vtraj_c = np.copy(vtraj[:,::4,:])
		allo.traj_s = np.copy(traj[:,3::4,:])
		allo.vtraj_s = np.copy(vtraj[:,3::4,:])
		allo.traj_sc = np.copy(traj[:,2::4,:])
		allo.vtraj_sc = np.copy(vtraj[:,2::4,:])
	if not has_bondinfo:
		return allo, data, cols, None, None
	if has_bondinfo:
		dist, engpot = read_dump_bondinfo(os.path.join(odir, 'bondinfo.dump'))
		return allo, data, cols, dist, engpot

def load_frame(odir, frame=200,train=True):
	if odir[-1] != '/' : odir += '/'
	netfile = glob.glob(odir+'*.txt')[0]
	datafile = glob.glob(odir+f'step{frame:d}.bond')[0]
	infile = glob.glob(odir+'*.in')[0]
	
	
	allo = Allosteric(netfile)
	dim = read_dim(infile)
	if dim != allo.dim:
		raise ValueError("Dimension mismatch between LAMMPS simulation (d={:d}) and network file (d={:d}).".format(dim,allo.dim))
	read_data_new(datafile, allo,train)
	return allo


def get_traj(odir,nframes=200,train=True):
	for i in range(1,nframes+1):
		allo = load_frame(odir, frame=i,train=train)
		stiffness = np.array([edge[2]['stiffness'] for edge in allo.graph.edges(data=True)])
		
		
		if i == 1:
			r_traj = np.copy(allo.pts)
			if train:
				rc_traj = np.copy(allo.pts_c)
				k_traj = np.copy(stiffness)
			else:
				rc_traj = None
				k_traj = None
		else:
			r_traj = np.concatenate((r_traj, allo.pts), axis=0)
			if train:
				rc_traj = np.concatenate((rc_traj, allo.pts_c), axis=0)
				k_traj = np.concatenate((k_traj, stiffness), axis=0)

	return r_traj, rc_traj, k_traj

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


