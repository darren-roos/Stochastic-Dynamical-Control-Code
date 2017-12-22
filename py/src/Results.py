# plt.plotting and results analysis module
import matplotlib as mpl, matplotlib.pyplot as plt, Ellipse

def plotTracking(ts, xs, ys, fmeans, us, obs, setpoint):

    tend = ts[-1]
    setpoints = numpy.ones(len(ts))*setpoint

    umax = max(abs(us))
    if umax == 0:
        subplt = 2
    else:
        subplt = 3
        
    mpl.rc("font", family="serif", serif="Computer Modern", size=12)
    mpl.rc("text", usetex=True)

    skipmeas = int(len(ts)/80)
    skipmean = int(len(ts)/40)
    plt.figure()
    plt.subplt.plot(subplt,1,1)
    x1, = plt.plot(ts, xs[0,:], "k", linewidth=3)
    if obs == 2: # plt.plot second measurement
        y2, = plt.plot(ts[::skipmeas], ys[0][::skipmeas], "kx", markersize=5, markeredgewidth=1)
        
    k1, = plt.plot(ts[::skipmean], fmeans[0][::skipmean], "bx", markersize=5, markeredgewidth = 2)
    ksp = plt.plot(ts, setpoints, "g-", linewidth=3)
    plt.ylabel(r"C$_A$ [kmol.m$^{-3}$]")
    plt.locator_params(nbins=4)
    plt.legend([x1, ksp],["Underlying model", "Set point"], loc="best", ncol=2)
    plt.xlim([0, tend])

    plt.subplt.plot(subplt,1,2)
    x2, = plt.plot(ts, xs[2,:], "k", linewidth=3)
    if obs == 1:
        y2, = plt.plot(ts[::skipmeas], ys[::skipmeas], "kx", markersize=5, markeredgewidth=1)
    else:
        y2, = plt.plot(ts[::skipmeas], ys[1][::skipmeas], "kx", markersize=5, markeredgewidth=1)
        
    k2, = plt.plot(ts[::skipmean], fmeans[1][::skipmean], "bx", markersize=5, markeredgewidth = 2)
    plt.ylabel(r"T$_R$ [K]")
    plt.locator_params(nbins=4)
    plt.legend([k2, y2],["Filtered mean", "Observations"], loc="best", ncol=2)
    plt.xlim([0, tend])
    # ylim([minimum(xs[2,:]), maximum(xs[2,:])])
    if subplt == 3:
        plt.subplt.plot(subplt,1,3)
        plt.plot(ts, (1/60.0)*us)
        plt.xlim([0, tend])
        plt.ylabel("Q [kW]")
        
    plt.locator_params(nbins=4)
    plt.xlabel("Time [min]")


def plotTracking(ts, xs, ys, fmeans, us, obs):

    tend = ts[-1]

    umax = max(abs(us))
    if umax == 0:
        subplt = 2
    else:
        subplt = 3
    
    mpl.rc("font", family="serif", serif="Computer Modern", size=12)
    mpl.rc("text", usetex=True)

    skipmeas = int(len(ts)/80)
    skipmean = int(len(ts)/40)
    plt.figure()
    plt.subplt.plot(subplt,1,1)
    x1, = plt.plot(ts, xs[1,:], "k", linewidth=3)
    if obs == 1: # plt.plot second measurement
        y2, = plt.plot(ts[::skipmeas], ys[0][::skipmeas], "kx", markersize=5, markeredgewidth=1)
    
    k1, = plt.plot(ts[::skipmean], fmeans[0][::skipmean], "bx", markersize=5, markeredgewidth = 2)
    plt.ylabel(r"C$_A$ [kmol.m$^{-3}$]")
    plt.locator_params(nbins=4)
    legend([x1],["Underlying model"], loc="best")
    plt.xlim([0, tend])
    # ylim([0, 1])

    plt.subplt.plot(subplt,1,2)
    x2, = plt.plot(ts, xs[1,:], "k", linewidth=3)
    if obs == 0:
        y2, = plt.plot(ts[::skipmeas], ys[::skipmeas], "kx", markersize=5, markeredgewidth=1)
    else:
        y2, = plt.plot(ts[::skipmeas], ys[1][::skipmeas][:], "kx", markersize=5, markeredgewidth=1)
    
    k2, = plt.plot(ts[::skipmean], fmeans[1][::skipmean], "bx", markersize=5, markeredgewidth = 2)
    plt.ylabel(r"T$_R$ [K]")
    plt.locator_params(nbins=4)
    legend([k2, y2],["Filtered mean", "Observations"], loc="best")
    plt.xlim([0, tend])
    # ylim([minimum(xs[2,:]), maximum(xs[2,:])])
    if subplt == 3:
        plt.subplt.plot(subplt,1,3)
        plt.plot(ts, (1.0/60.0)*us)
        plt.xlim([0, tend])
        plt.ylabel("Q [kW]")
    
    plt.locator_params(nbins=4)
    plt.xlabel("Time [min]")
    


def plotStateSpaceSwitch(linsystems, xs):
    mpl.rc("font", family="serif", serif="Computer Modern", size=12)
    mpl.rc("text", usetex=True)
    plt.figure() # Model and state space
    for k in range(len(linsystems)):
        plt.plot(linsystems[k].op[0],linsystems[k].op[1],"kx",markersize=5, markeredgewidth=1)
        annotate(latexstring("M_",k),
            xy=[linsystems[k].op[0],linsystems[k].op[1]],
            xytext=[linsystems[k].op[0],linsystems[k].op[1]],
            fontsize=12.0,
            ha="center",
            va="bottom")
        
    plt.plot(xs[0], xs[1], "k", linewidth=3)
    plt.plot(xs[0][0], xs[1][0], "ko", markersize=10, markeredgewidth = 4)
    plt.plot(xs[0][-1], xs[1][-1], "kx", markersize=10, markeredgewidth = 4)
    plt.xlim([-0.1, 1.1])
    plt.xlabel(r"C$_A$ [kmol.m$^{-3}$]")
    plt.ylabel(r"T$_R$ [K]")
    


def plotSwitchSelection(numSwitches, strack, ts, cbaron):

    plt.figure() # Model selection
    mpl.rc("font", family="serif", serif="Computer Modern", size=12)
    mpl.rc("text", usetex=True)
    axes = [None]*numSwitches
    im = 0
    width = 500
    for k in range(numSwitches):
        ax = plt.subplot(numSwitches, 1, k)
        axes[k] = ax
        if cbaron:
            im = plt.imshow(repeat(strack[k,:], outer=[width, 1]), cmap="cubehelix_r",vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
        else:
            im = plt.imshow(repeat(strack[k,:], outer=[width, 1]), cmap="binary",vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
        
        plt.tick_params(axis="y", which="both",left="off",right="off", labelleft = "off")
        plt.tick_params(axis="x", which="both",bottom="off", labelbottom = "off")
        plt.ylabel(latexstring("M_",k))
    

    plt.tick_params(axis="x", labelbottom = "on")
    tempts = range(0, len(ts), int(len(ts)/10.0))
    temp = [None]*len(tempts)
    for lat in range(len(tempts)):
        temp[lat] = latexstring(ts[tempts[lat]])
    

    plt.xticks(tempts, temp)

    if cbaron == True:
        plt.colorbar(im, ax=axes)
    
    plt.xlabel("Time [min]")


def plotEllipses(ts, xs, fmeans, fcovars, fname, legloc):

    mpl.rc("font", family="serif", serif="Computer Modern", size=12)
    mpl.rc("text", usetex=True)
    N = len(ts)
    skip = int(len(ts)/40)
    plt.figure()
    b1 = 0.0
    for k in range(N):
        p1, p2 = Ellipse.ellipse(fmeans[:,k], fcovars[:,:, k])
        # b1, = plt.plot(p1, p2, "b")
        b1, = plt.fill(p1, p2, "b", edgecolor="none")
    
    x1, = plt.plot(xs[0,:], xs[1,:], "k",linewidth=3)
    f1, = plt.plot(fmeans[0][::skip], fmeans[1][::skip], "mx", markersize=5, markeredgewidth = 2)
    plt.plot(xs[0,0], xs[1,0], "ko", markersize=10, markeredgewidth = 4)
    plt.plot(xs[0,-1], xs[1,-1], "kx", markersize=10, markeredgewidth = 4)
    plt.ylabel(r"T$_R$ [K]")
    plt.xlabel(r"C$_A$ [kmol.m$^{-3}$]")
    plt.legend([x1,f1, b1],["Underlying model","Filtered mean", r"90$\%$ Confidence region"], loc=legloc)
    

"""
def plotEllipses(ts, xs, fmeans, fcovars, fname, line, sp, nf, sigma, pick, legloc)

  mpl.rc("font", family="serif", serif="Computer Modern", size=12)
  mpl.rc("text", usetex=True)
  N = len(ts)
  skip = int(len(ts)/40)

  nf && plt.figure() # only create a new plt.figure if required
  b1 = 0.0
  for k=1:N
    p1, p2 = Ellipse.ellipse(fmeans[:,k], fcovars[:,:, k], sigma)
    # b1, = plt.plot(p1, p2, "b")
    b1, = fill(p1, p2, "b", edgecolor="none")
  end
  x1, = plt.plot(xs[1,:][:], xs[2,:][:], "k",linewidth=3)
  f1, = plt.plot(fmeans[1, 1:skip:end][:], fmeans[2, 1:skip:end][:], "mx", markersize=5, markeredgewidth = 2)
  #plt.plot(xs[1, 1:skip:end][:], xs[2, 1:skip:end][:], "kx", markersize=5, markeredgewidth = 2)
  plt.plot(xs[1,1], xs[2,1], "ko", markersize=10, markeredgewidth = 4)
  plt.plot(xs[1,end], xs[2,end], "kx", markersize=10, markeredgewidth = 4)

  # line = [b,c] => y + bx + c = 0
  # line => y = - bx - c

  lxs = [-0.1:0.05:1.1]
  lys = -line[1].*lxs .- line[2]
  plt.xlim([0.0, 1.0])
  ylim([minimum(xs[2,:]-10), maximum(xs[2, :]+10)])
  plt.plot(lxs, lys, "r-")

  plt.plot(sp[1], sp[2], "gx",markersize=8, markeredgewidth = 4)

  plt.ylabel(r"T$_R$ [K]")
  plt.xlabel(r"C$_A$ [kmol.m$^{-3}$]")
  # conf = round((1.0 - exp(-sigma/2.0))*100.0, 3)
  # temp = latexstring(conf,"\%", "Confidence~Region")
  # legend([x1,f1, b1],[r"Underlying~Model",r"Filtered~Mean", temp], loc="best")
  if pick==1
    legend([x1,f1, b1],["Underlying model","Filtered mean", r"90$\%$ Confidence region"], loc=legloc)
  elseif pick == 2
    legend([x1,f1, b1],["Underlying model","Filtered mean", r"99$\%$ Confidence region"], loc=legloc)
  elseif pick == 3
    legend([x1,f1, b1],["Underlying model","Filtered mean", r"99.9$\%$ Confidence region"], loc=legloc)
  else
    legend([x1,f1, b1],["Underlying model","Filtered mean", r"99.99$\%$ Confidence region"], loc=legloc)
  end
end

def plotEllipseComp(f1means, f1covars, f2means, f2covars, xs, ts, sigma=4.605)

  N = len(ts)
  skip = int(len(ts)/30)
  plt.figure()
  mpl.rc("font", family="serif", serif="Computer Modern", size=12)
  mpl.rc("text", usetex=True)
  x1, = plt.plot(xs[1,:][:], xs[2,:][:], "k",linewidth=3)
  f1, = plt.plot(f1means[1, 1:skip:end][:], f1means[2, 1:skip:end][:], "yx", markersize=5, markeredgewidth = 2)
  f2, = plt.plot(f2means[1, 1:skip:end][:], f2means[2, 1:skip:end][:], "gx", markersize=5, markeredgewidth = 2)
  b1 = 0.0
  b2 = 0.0
  for k=1:N
    p1, p2 = Ellipse.ellipse(f1means[:,k], f1covars[:,:, k], sigma)
    # b1, = plt.plot(p1, p2, "r")
    b1, = fill(p1, p2, "r", edgecolor="none")

    p3, p4 = Ellipse.ellipse(f2means[:,k], f2covars[:,:, k], sigma)
    # b2, = plt.plot(p3, p4, "b")
    b2, = fill(p3, p4, "b", edgecolor="none")
  end

  plt.plot(xs[1,1], xs[2,1], "ko", markersize=10, markeredgewidth = 4)
  plt.plot(xs[1,end], xs[2,end], "kx", markersize=10, markeredgewidth = 4)
  plt.ylabel(r"T$_R$ [K]")
  plt.xlabel(r"C$_A$ [kmol.m$^{-3}$]")
  legend([x1,f1,f2, b1, b2],["Underlying model","Particle filter","Kalman filter", r"PF 90$\%$ Confidence region", r"KF 90$\%$ Confidence region"], loc="best")
end

def plotTrackingBreak(ts, xs, xsb, ys, fmeans, obs)

  N = len(ts)
  tend = ts[end]
  skipm = int(len(ts)/80)
  plt.figure() # plt.plot filtered results
  mpl.rc("font", family="serif", serif="Computer Modern", size=12)
  mpl.rc("text", usetex=True)
  plt.subplt.plot(2,1,1)
  x1, = plt.plot(ts, xs[1,:], "k", linewidth=3)
  x1nf, = plt.plot(ts, xsb[1,:], "g--", linewidth=3)
  if obs == 2
    y2, = plt.plot(ts[1:skipm:end], ys[1, 1:skipm:end][:], "kx", markersize=5, markeredgewidth=1)
  end
  k1, = plt.plot(ts, fmeans[1,:], "r--", linewidth=3)
  plt.ylabel(r"C$_A$ [kmol.m$^{-3}$]")
  plt.locator_params(nbins=4)
  legend([x1, k1],["Underlying model","Filtered mean"], loc="best")
  plt.xlim([0, tend])
  plt.subplt.plot(2,1,2)
  x2, = plt.plot(ts, xs[2,:], "k", linewidth=3)
  x2nf, = plt.plot(ts, xsb[2,:], "g--", linewidth=3)
  if obs == 2
    y2, = plt.plot(ts[1:skipm:end], ys[2, 1:skipm:end][:], "kx", markersize=5, markeredgewidth=1)
  else
    y2, = plt.plot(ts[1:skipm:end], ys[1:skipm:end], "kx", markersize=5, markeredgewidth=1)
  end
  k2, = plt.plot(ts, fmeans[2,:], "r--", linewidth=3)
  plt.ylabel(r"T$_R$ [K]")
  plt.locator_params(nbins=4)
  plt.xlabel("Time [min]")
  legend([y2, x2nf],["Observations","Underlying model w/o fault"], loc="best")
  plt.xlim([0, tend])
end

def plotTrackingTwoFilters(ts, xs, ys, f1means, f2means)

  skipm = int(len(ts)/80)
  skip = int(len(ts)/40)
  tend = ts[end]
  plt.figure() # plt.plot filtered results
  mpl.rc("font", family="serif", serif="Computer Modern", size=12)
  mpl.rc("text", usetex=True)
  plt.subplt.plot(2,1,1)
  x1, = plt.plot(ts, xs[1,:], "k", linewidth=3)
  k1, = plt.plot(ts[1:skip:end], f1means[1,1:skip:end], "rx", markersize=5, markeredgewidth=2)
  y2, = plt.plot(ts[1:skipm:end], ys[1, 1:skipm:end][:], "kx", markersize=5, markeredgewidth=1)
  k12, = plt.plot(ts[1:skip:end], f2means[1, 1:skip:end], "bx", markersize=5, markeredgewidth=2)
  plt.ylabel(r"C$_A$ [kmol.m$^{-3}$]")
  legend([x1, k1],["Underlying model","Particle filter"], loc="best", ncol=2)
  plt.xlim([0, tend])
  plt.subplt.plot(2,1,2)
  x2, = plt.plot(ts, xs[2,:], "k", linewidth=3)
  y2, = plt.plot(ts[1:skipm:end], ys[2, 1:skipm:end][:], "kx", markersize=5, markeredgewidth=1)
  k2, = plt.plot(ts[1:skip:end], f1means[2,1:skip:end], "rx", markersize=5, markeredgewidth=2)
  k22, = plt.plot(ts[1:skip:end], f2means[2, 1:skip:end], "bx", markersize=5, markeredgewidth=2)
  plt.ylabel(r"T$_R$ [K]")
  plt.xlabel("Time [min]")
  legend([y2, k22],["Observations", "Kalman filter"], loc="best", ncol=2)
  plt.xlim([0, tend])
end

def plotKLdiv(ts, kldiv, basediv, unidiv, logged)
  mpl.rc("font", family="serif", serif="Computer Modern", size=12)
  mpl.rc("text", usetex=True)

  plt.figure()
  if logged
    kl, = semilogy(ts, kldiv, "r", linewidth=3)
    gd, = semilogy(ts, basediv, "b", linewidth=3)
    ud, = semilogy(ts, unidiv, "g", linewidth=3)
  else
    kl, = plt.plot(ts, kldiv, "r", linewidth=3)
    gd, = plt.plot(ts, basediv, "b", linewidth=3)
    ud, = plt.plot(ts, unidiv, "g", linewidth=3)
  end
  plt.xlabel("Time [min]")
  plt.ylabel("Divergence [Nats]")
  legend([kl, gd, ud],["Approximation","Baseline", "Uniform"], loc="best")
end

def calcError(x, y::Array{Float64, 2})

  r, N = size(x)
  avediff1 = (1.0/N)*sum(abs((x[1, :].-y[1, :])./x[1,:]))*100.0
  avediff2 = (1.0/N)*sum(abs((x[2, :].-y[2, :])./x[2,:]))*100.0

  print("Average Concentration Error: ", round(avediff1, 4),  "%")
  print("Average Temperature Error: ", round(avediff2, 4), "%")
  return avediff1, avediff2
end

def calcError(x, y::Float64)

  r, N = size(x)
  avediff1 = (1.0/N)*sum(abs((x[1, :] - y)./y))*100.0

  print("Average Concentration Error: ", round(avediff1, 4),  "%")
  return avediff1
end

def calcError2(x, y::Float64)

  r, N = size(x)
  avediff1 = (1.0/N)*sum(abs((x[1, :] - y)./y))*100.0

  return avediff1
end

def calcError3(x, y::Float64)

  N = len(x)
  avediff1 = (1.0/N)*sum(abs((x .- y)./y))*100.0

  return avediff1
end

def calcEnergy(us, uss, h)
  N = len(us)
  avecost = (1./(60.0*N))*sum(abs(us-uss))
  print("Average Input (kW): ", avecost)
  return avecost
end

def checkConstraint(ts, xs, line)
  # line = [b,c] => y + bx + c = 0
  # line => y = - bx - c
  r, N = size(xs)
  conmargin = zeros(N)
  minneg = Inf
  minpos = Inf
  for k=1:N
    temp = xs[2, k] + xs[1, k]*line[1] + line[2]
    if temp < 0.0
      conmargin[k] = -abs(temp)/sqrt(line[1]^2 + 1.0)
      if minneg > abs(temp)/sqrt(line[1]^2 + 1.0)
        minneg = abs(temp)/sqrt(line[1]^2 + 1.0)
      end
    else
      conmargin[k] = abs(temp)/sqrt(line[1]^2 + 1.0)
      if minpos > abs(temp)/sqrt(line[1]^2 + 1.0)
        minpos = abs(temp)/sqrt(line[1]^2 + 1.0)
      end
    end
  end


  print("Minimum Positive Clearance: ", minpos)
  print("Minimum Negative Clearance: ", minneg)

  plt.figure()
  mpl.rc("font", family="serif", size=12)
  mpl.rc("text", usetex=True)

  plt.plot(ts, zeros(N), "r", linewidth=1)
  plt.plot(ts, conmargin, "k", linewidth=3)
  plt.xlabel(r"Time [min]")
  plt.ylabel(r"Clearance")
end

def getMCRes!(xs, sigmas, line, mcdistmat, counter, h)
  # line = [b,c] => y + bx + c = 0
  # line => y = - bx - c
  d = [line[1], 1.0]
  r, N = size(xs)
  negdist = 0.0
  timeviolated = 0
  for k=1:N
    temp = xs[2, k] + xs[1, k]*line[1] + line[2] # check constraint
    if temp < 0.0
      negdist += -abs(temp)/sqrt(d'*sigmas[:,:, k]*d)[1]
      timeviolated += 1
    end
  end
  mcdistmat[1, counter] = negdist*h # area integral
  mcdistmat[2,counter] = timeviolated*h # in minutes
end

end #module"""
