def function1(x, gamma, beta, eps):

  N, D = x.shape


  mu = 1./N * np.sum(x, axis = 0)

 
  xmu = x - mu

  
  sq = xmu ** 2


  var = 1./N * np.sum(sq, axis = 0)

  
  sqrtvar = np.sqrt(var + eps)

  ivar = 1./sqrtvar

  
  xhat = xmu * ivar

  
  gammax = gamma * xhat


  out = gammax + beta

  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache







def function2(dout, cache):

  
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache


  N,D = dout.shape

  
  dbeta = np.sum(dout, axis=0)
  dgammax = dout 


  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma


  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar


  dsqrtvar = -1. /(sqrtvar**2) * divar


  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar


  dsq = 1. /N * np.ones((N,D)) * dvar


  dxmu2 = 2 * xmu * dsq


  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)


  dx2 = 1. /N * np.ones((N,D)) * dmu


  dx = dx1 + dx2

  return dx, dgamma, dbeta