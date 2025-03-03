from scipy.special import gammaln

def log_Beta(a):
	z = sum(gammaln(a)) - gammaln(sum(a))
	return z