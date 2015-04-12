# Qualitative Analysis of the CSTR

using NLsolve
using Reactor

# First, lets inspect the steady state values for different parameters
cstr1 = begin
  V = 5.0 #m3
  R = 8.314 #kJ/kmol.K
  CA0 = 1.0 #kmol/m3
  TA0 = 310.0 #K
  dH = -4.78e4 #kJ/kmol
  k0 = 72.0e7 #1/min
  E = 8.314e4 #kJ/kmol
  Cp = 0.239 #kJ/kgK
  rho = 1000.0 #kg/m3
  F = 100e-3 #m3/min
  Reactor.reactor(V, R, CA0, TA0, dH, k0, E, Cp, rho, F)
end

N = 100
Ts = linspace(200, 600, N) # temperature range
qrs1 = zeros(N) # heat removals
qrs2 = zeros(N) # heat removals
qrs3 = zeros(N) # heat removals
qgs1 = zeros(N) # heat generations

for k=1:N
  qrs1[k] = Reactor.QR(Ts[k], -906.0, cstr1)
  qrs2[k] = Reactor.QR(Ts[k], 0.0, cstr1)
  qrs3[k] = Reactor.QR(Ts[k], 1145.0, cstr1)
  qgs1[k] = Reactor.QG(Ts[k], cstr1)
end

rc("font", family="serif", size=24)
figure(1)
q1, = plot(Ts, qrs1, "b", linewidth=3)
q2, = plot(Ts, qrs2, "g", linewidth=3)
q3, = plot(Ts, qrs3, "r", linewidth=3)
opline1, = plot(Ts, qgs1, "k", linewidth=3)
legend([q1,q2,q3, opline1],["Q=-906 kJ/min","Q=0 kJ/min","Q=1145 kJ/min","Operating Curve"], loc="best")
xlabel("Steady State Temperature [K]")
ylabel("Heat Removal Rate [K/min]")
ylim([0.0, 5.])

#Get the steady state points
xguess1 = [0.073, 493.0]
xguess2 = [0.21, 467.0]
xguess3 = [0.999, 310.0]
f!(x, xvec) = Reactor.reactor_func!(x, 0.0, cstr1, xvec)

xx1res = nlsolve(f!, xguess1)
writecsv("ss1.csv", xx1res.zero)
xx2res = nlsolve(f!, xguess2)
writecsv("ss2.csv", xx2res.zero)
xx3res = nlsolve(f!, xguess3)
writecsv("ss3.csv", xx3res.zero)

println("Nominal Operating Points")
println("High Heat: ", xx1res.zero)
println("Eigenvalues: ", eig(Reactor.jacobian(xx1res.zero, cstr1))[1])
println("Medium Heat: ", xx2res.zero)
println("Eigenvalues: ", eig(Reactor.jacobian(xx2res.zero, cstr1))[1])
println("Low Heat: ", xx3res.zero)
println("Eigenvalues: ", eig(Reactor.jacobian(xx3res.zero, cstr1))[1])


## Get the bifurcation points
# Low heat input
xguess1 = [0.999, 270.0]
xguess2 = [0.11, 450.0]
flag = true
Q = -800.0
prevss1 = zeros(2)
prevss2 = zeros(2)
while flag
  f!(x, xvec) = Reactor.reactor_func!(x, Q, cstr1, xvec)
  xx1res = nlsolve(f!, xguess1)
  xx2res = nlsolve(f!, xguess2)
  flag = converged(xx2res)
  if flag
    prevss1 = xx1res.zero
    prevss2 = xx2res.zero
  end
  Q = Q - 1.0
end
println("Low heat: ", prevss1)
println("Eigenvalues: ", eig(Reactor.jacobian(prevss1, cstr1))[1])
println("*****")
println("Low heat: ", prevss2)
println("Eigenvalues: ", eig(Reactor.jacobian(prevss2, cstr1))[1])
println("*****")
# High heat input
xguess1 = [0.93, 370.0]
xguess2 = [0.0011, 570.0]
flag = true
Q = 1100.0
while flag
  f!(x, xvec) = Reactor.reactor_func!(x, Q, cstr1, xvec)
  xx1res = nlsolve(f!, xguess1)
  xx2res = nlsolve(f!, xguess2)
  flag = converged(xx1res)
  if flag
    prevss1 = xx1res.zero
    prevss2 = xx2res.zero
  end
  Q = Q + 1.0
  if Q > 1300.0
    flag = false
  end
end
println("High heat: ", prevss1)
println("Eigenvalues: ", eig(Reactor.jacobian(prevss1, cstr1))[1])
println("*****")
println("High heat: ", prevss2)
println("Eigenvalues: ", eig(Reactor.jacobian(prevss2, cstr1))[1])
println("*****")