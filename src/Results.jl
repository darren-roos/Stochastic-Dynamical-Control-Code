# Plotting and results analysis module
module Results

using PyPlot
using Ellipse

function plotTracking(ts, xs, ys, fmeans, us, obs)

  tend = ts[end]

  umax = maximum(abs(us))
  if umax == 0.0
    subplt = 2
  else
    subplt = 3
  end
  rc("font", family="serif", size=24)

  skipmeas = int(length(ts)/20)
  skipmean = int(length(ts)/20)
  figure()
  subplot(subplt,1,1)
  x1, = plot(ts, xs[1,:]', "k", linewidth=3)
  if obs == 2 # plot second measurement
    y2, = plot(ts[1:skipmeas:end], ys[1, 1:skipmeas:end][:], "kx", markersize=5, markeredgewidth=1)
  end
  k1, = plot(ts[1:skipmean:end], fmeans[1, 1:skipmean:end]', "bx", markersize=5, markeredgewidth = 2)
  ylabel(L"Concentration [kmol.m$^{-3}$]")
  legend([x1],["Nonlinear Model"], loc="best")
  xlim([0, tend])
  ylim([0, 1])
  subplot(subplt,1,2)
  x2, = plot(ts, xs[2,:]', "k", linewidth=3)
  if obs == 1
    y2, = plot(ts[1:skipmeas:end], ys[1:skipmeas:end], "kx", markersize=5, markeredgewidth=1)
  else
    y2, = plot(ts[1:skipmeas:end], ys[2, 1:skipmeas:end][:], "kx", markersize=5, markeredgewidth=1)
  end
  k2, = plot(ts[1:skipmean:end], fmeans[2, 1:skipmean:end]', "bx", markersize=5, markeredgewidth = 2)
  ylabel("Temperature [K]")
  legend([k2, y2],["Filtered Mean Estimate", "Nonlinear Model Measured"], loc="best")
  xlim([0, tend])
  ylim([minimum(xs[2,:]), maximum(xs[2,:])])
  if subplt == 3
    subplot(subplt,1,3)
    plot(ts, us)
    xlim([0, tend])
    ylabel("Controller Input")
  end
  xlabel("Time [min]")
end

function plotStateSpaceSwitch(linsystems, xs)

  rc("font", family="serif", size=24)

  figure() # Model and state space
  for k=1:length(linsystems)
    plot(linsystems[k].op[1],linsystems[k].op[2],"kx",markersize=5, markeredgewidth=1)
  annotate(string("Switch: ", k),
        xy=[linsystems[k].op[1],linsystems[k].op[2]],
        xytext=[linsystems[k].op[1],linsystems[k].op[2]],
        fontsize=22.0,
        ha="center",
        va="bottom")
  end
  plot(xs[1,:][:], xs[2,:][:], "k", linewidth=3)
  plot(xs[1,1], xs[2,1], "ko", markersize=10, markeredgewidth = 4)
  plot(xs[1,end], xs[2,end], "kx", markersize=10, markeredgewidth = 4)
  xlim([-0.1, 1.1])
  xlabel(L"Concentration [kmol.m$^{-3}$]")
  ylabel("Temperature [K]")
end

function plotSwitchSelection(numSwitches, strack, ts, cbaron)
  figure() # Model selection
  axes = Array(Any, numSwitches)
  im = 0
  width = 500
  for k=1:numSwitches
    ax = subplot(numSwitches, 1, k)
    axes[k] = ax
    im = imshow(repeat(strack[k,:], outer=[width, 1]), cmap="cubehelix",vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    tick_params(axis="y", which="both",left="off",right="off", labelleft = "off")
    tick_params(axis="x", which="both",bottom="off", labelbottom = "off")
    ylabel(string("S::",k))
  end

  tick_params(axis="x", labelbottom = "on")
  xticks([1:int(length(ts)/10.0):length(ts)], ts[1:int(length(ts)/10.0):end])

  if cbaron == true
    colorbar(im, ax=axes)
  end
  xlabel("Time [min]")
end

function plotEllipses(ts, xs, ys, fmeans, fcovars)

  rc("font", family="serif", size=24)
  N = length(ts)
  skip = int(length(ts)/20)
  figure(1)
  x1, = plot(xs[1,:][:], xs[2,:][:], "k",linewidth=3)
  f1, = plot(fmeans[1, 1:skip:end][:], fmeans[2, 1:skip:end][:], "bx", markersize=5, markeredgewidth = 2)
  b1 = 0.0
  for k=1:skip:N
    p1, p2 = Ellipse.ellipse(fmeans[:,k], fcovars[:,:, k])
    b1, = plot(p1, p2, "b")
  end
  plot(xs[1, 1:skip:end][:], xs[2, 1:skip:end][:], "kx", markersize=5, markeredgewidth = 2)
  plot(xs[1,1], xs[2,1], "ko", markersize=10, markeredgewidth = 4)
  plot(xs[1,end], xs[2,end], "kx", markersize=10, markeredgewidth = 4)
  ylabel("Temperature [K]")
  xlabel(L"Concentration [kmol.m$^{-3}$]")
  legend([x1,f1, b1],["Nonlinear Model","Kalman Filter Mean", L"Kalman Filter $1\sigma$-Ellipse"], loc="best")
end


end #module