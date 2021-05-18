import plotly.graph_objects as go
import numpy as np
from math import gamma

def set_basic_layout(fig):
	fig.update_xaxes(gridcolor="gray",zeroline=False)
	fig.update_xaxes(ticks="inside",tickwidth=3,ticklen=5,tickcolor="firebrick")
	fig.update_yaxes(gridcolor="gray",zeroline=False)
	fig.update_yaxes(ticks="inside",tickwidth=3,ticklen=5,tickcolor="firebrick")
	fig.update_layout(margin=dict(t=10,l=10,b=10,r=10))
	fig.update_layout(font=dict(family="Times",size=18))
	return fig

def plot_1D_Gaussian(mean,sigma):
	left = mean-sigma*3
	right = mean+sigma*3
	X = np.linspace(left,right,300)
	fig = go.Figure()
	pdf = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(1/sigma**2)*(X-mean)**2)
	fig.add_trace(go.Scatter(x=X, 
							 y=pdf,
							 mode="lines", 
							 showlegend=False,
							 line=dict(color="crimson",width=3)))

	fig.update_layout(xaxis_title="X",yaxis_title="pdf")				 
	fig = set_basic_layout(fig)
	return fig

def plot_1D_MoG(means,sigmas,weights):
	assert np.isclose(np.sum(weights),1), "mixing weights must sum to 1"
	lefts = np.array(means)-np.array(sigmas)*3
	rights = np.array(means)+np.array(sigmas)*3
	X = np.linspace(np.min(lefts),np.max(rights),300)
	total_prob = np.zeros(len(X))
	fig = go.Figure()
	for i in range(len(means)):
		pdf = weights[i]*(1/(np.sqrt(2*np.pi)*sigmas[i]))*np.exp(-(1/sigmas[i]**2)*(X-means[i])**2)
		total_prob+=pdf
		fig.add_trace(go.Scatter(x=X, 
							 y=pdf,
							 name="element "+str(i+1), 
							 mode="lines", 
							 showlegend=True,
							 line=dict(width=2)))
	fig.add_trace(go.Scatter(x=X, 
							 y=total_prob,
							 name="joint", 
							 mode="lines", 
							 showlegend=True,
							 line=dict(color="black",width=3)))

	fig.update_layout(xaxis_title="X",yaxis_title="pdf")				 
	fig = set_basic_layout(fig)
	return fig
	
def plot_binomial(theta,n):
	k = np.arange(n)
	n_choose_k = np.math.factorial(n)/np.array([np.math.factorial(k_i)*np.math.factorial(n-k_i) for k_i in k])
	prob_k = n_choose_k * theta**k * (1-theta)**(n-k)
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=k, 
							 y=prob_k,
							 name="n={} theta={}".format(n,theta), 
							 mode="lines+markers", 
							 showlegend=True,
							 line=dict(width=2),
							 marker=dict(size=5)))
	fig.update_layout(xaxis_title="X",yaxis_title="pmf")				 
	fig = set_basic_layout(fig)
	fig.update_xaxes(tickmode="auto",nticks=min(n,15))
	return fig
	
def plot_beta_model(alpha,beta):
	x = np.linspace(0,1,100)
	unnorm_prob = x**(alpha-1)*(1-x)**(beta-1)
	norm_prob = unnorm_prob/((gamma(alpha)*gamma(beta))/gamma(alpha+beta))
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=x, 
							 y=norm_prob,
							 name="alpha={} beta={}".format(alpha,beta), 
							 mode="lines+markers", 
							 showlegend=True,
							 line=dict(width=2),
							 marker=dict(size=5)))
	fig.update_layout(xaxis_title="X",yaxis_title="pdf")				 
	fig = set_basic_layout(fig)
	fig.update_xaxes(tickmode="auto",nticks=10)
	return fig

def plot_2D_Gaussian_Contour(mean,cov):	
	def gaussian_pdf(x,mean,cov,prec=None):
		if(prec==None):
		    prec = np.linalg.inv(cov)
		norm = (np.sqrt(2*np.pi)**(len(mean)))*np.sqrt(np.linalg.det(cov))
		x_dif = (x-mean).reshape(x.shape[0],len(mean))
		exponent = np.exp(-0.5*np.sum(x_dif.dot(prec)*x_dif,axis=1))
		return exponent/norm

	std = np.sqrt(np.sum(cov,axis=0))
	grid_size=100
	grid_lower_lim = mean-std*3
	grid_upper_lim = mean+std*3
	steps = std*6/grid_size
	grid = np.mgrid[grid_lower_lim[0]:grid_upper_lim[0]:steps[0],grid_lower_lim[1]:grid_upper_lim[1]:steps[1]]
	grid = grid.reshape(2,-1).T
	z = gaussian_pdf(grid,mean,cov)
	x = np.linspace(grid_lower_lim[0],grid_upper_lim[0],grid_size)
	y = np.linspace(grid_lower_lim[1],grid_upper_lim[1],grid_size)
	fig = go.Figure()
	fig.add_trace(go.Contour(
			z=z.reshape(grid_size,grid_size),
			x=x,
			y=y,
			colorscale="hot",
			showscale=True))
	fig = set_basic_layout(fig)
	fig.update_layout(font=dict(size=16))
	return fig

def plot_2D_Hist(samples,nbins):
	std = np.std(samples,axis=0)
	fig = go.Figure()
	fig.add_trace(go.Histogram2d(
		    x=samples[:,0],
		    y=samples[:,1],
		    xbins=dict(size=std[0]*6/nbins),
		    ybins=dict(size=std[1]*6/nbins),
		    colorscale="hot",
		    showscale=True))

	fig = set_basic_layout(fig)
	return fig



