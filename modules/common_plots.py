import plotly.graph_objects as go
import numpy as np

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
	
