import os

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2016/bin/x86_64-darwin/'

import numpy
import matplotlib.pyplot as plt
from joblib.memory import Memory

from matplotlib import rc
import matplotlib

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{siunitx}\sisetup{detect-weight=true, detect-family=true}"]
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}\usepackage{siunitx}\sisetup{detect-weight=true, detect-family=true}"]


memory = Memory(cachedir="plotsperformance", verbose=0, compress=9)

def loadtest(folder):
    result = []
    for fold in range(10):
        result.append(numpy.load(folder + "/fold_" + str(fold) + "_test.npz")['arr_0'])
    # return numpy.ravel(numpy.array(result))
    return result

@memory.cache
def loadLL(folder):
    result = []
    for fold in range(10):
        shape = numpy.load(folder + "/fold_" + str(fold) + "_full_train.npz")['arr_0'].shape
        a = numpy.memmap(folder + "/fold_" + str(fold) + "_fulltrainll.mmp", dtype='float32', mode='r', shape=shape)
        
        a = numpy.sum(a)
        a = -a
        print(folder, a)
        result.append(a)
    # return numpy.ravel(numpy.array(result))
    return result

def loadpred(folder):
    result = []
    for fold in range(10):
        shape = numpy.load(folder + "/fold_" + str(fold) + "_test.npz")['arr_0'].shape
        a = numpy.memmap(folder + "/fold_" + str(fold) + "_pred.mmp", dtype='float32', mode='r', shape=shape)
        result.append(a)
    # return numpy.ravel(numpy.array(result))
    return result

#a = loadLL("data/TrafficLarge/cs/10")
#print(a)


@memory.cache
def loadtimes(folder):
    shape = numpy.load(folder + "/fold_0_test.npz")['arr_0'].shape
    result = []
    for fold in range(10):
        result.append(numpy.sum(numpy.memmap(folder + "/fold_" + str(fold) + "_time.mmp", dtype='float32', mode='r', shape=(shape[1], 1))))
    # return numpy.ravel(numpy.array(result))
    return numpy.ravel(result)


@memory.cache
def mse(folder, clipped=False, floored=False):
    test = loadtest(folder)

    pred = loadpred(folder)

    result = []
    for i in range(10):

        p = pred[i]

        if clipped:
            p = numpy.clip(p, 0, 1)
        if floored:
            p = numpy.floor(p)

        err = numpy.power(test[i] - p, 2)
        mse = numpy.mean(err)
        rmse = numpy.sqrt(mse)
        result.append(rmse)

    # assert test.shape == pred.shape
    print(folder, result)
    return result

        


def boxplot(ax, data, positions, color):
    meanlineprops = dict(linestyle='-', linewidth=3)

    bp = ax.boxplot(data, positions=positions, showfliers=False, widths=0.7, medianprops=meanlineprops)
    
    for element in ["boxes", "caps", "whiskers", "fliers"]:
        for obj in bp[element]:
            plt.setp(obj, color="black")
            
    for obj in bp["medians"]:
        plt.setp(obj, color=color)


red = "#D32726"
blue = "#0160B6"
black = "#000000"

plt.rc('grid', linestyle="--")
plt.rcParams["font.weight"] = "bold"

fig = plt.figure(1, figsize=(6.4, 3.5))

def clog(lst):
    res = []
    for l in lst:
        res.append(numpy.log(l))
    return res


######################## MNIST LL


cs = [
      loadLL("data/MNIST/cs/10"),
      loadLL("data/MNIST/cs/20"),
      loadLL("data/MNIST/cs/30"),
      loadLL("data/MNIST/cs/40"),
      ]

rnd = [
      loadLL("data/MNIST/rnd/10"),
      loadLL("data/MNIST/rnd/20"),
      loadLL("data/MNIST/rnd/30"),
      loadLL("data/MNIST/rnd/40"),
      ]


full = [loadLL("data/MNIST/full")
        ]

print("empirical eps MNIST")


for i in range(len(cs)):
    eepscs = []
    eepsrnd = []
    for fold in range(10):
        eepscs.append(numpy.abs(cs[i][fold] - full[0][fold]) / full[0][fold])
        eepsrnd.append(numpy.abs(rnd[i][fold] - full[0][fold]) / full[0][fold])
    print(numpy.round(numpy.mean(eepscs)*100,2), numpy.round(numpy.mean(eepsrnd)*100,2))



cs = list(map(lambda t: numpy.array(t) / 10000000,cs))
rnd = list(map(lambda t: numpy.array(t) / 10000000,rnd))
full = list(map(lambda t: numpy.array(t) / 10000000,full))

print(cs)

print(full)

# Create an axes instance
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.18, bottom=0.13, right=0.999, top=0.90, wspace=0, hspace=0)



# Create the boxplot
boxplot(ax, cs, positions=list(range(1,len(cs)*2,2)), color=blue)

boxplot(ax, rnd, positions=list(range(2,len(rnd)*2+1,2)), color=red)

boxplot(ax, full, positions=[len(cs)+len(rnd)+1], color=black)

ax.set_xlim([0, 10])
ax.set_ylim([3.55, 3.75])
ax.yaxis.grid()

ax.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (s), ['10\%', '20\%', '30\%', '40\%', "100\%"]), weight="bold", fontsize=12)
ax.set_xticks([1.5, 3.5, 5.5, 7.5, 9])

ytickspos = numpy.arange(3.55,3.80,0.05)
ax.set_yticklabels(map(lambda s: r"$\bm{%s\!\!\times\!\!10^7}$" % (str(s)), ytickspos), weight="bold", fontsize=12)
ax.set_yticks(ytickspos)


#ax.yaxis.set_tick_params(weight="bold")

plt.axvline(x=2.5, color=blue)
plt.axvline(x=6.5, color=red)


plt.ylabel(r'\textbf{Negative Gaussian Pseudo Log--Likelihood}',fontweight='bold')
plt.xlabel(r'\textbf{Training data (Sample size in percentage)}',fontweight='bold')




ax.get_xaxis().tick_bottom()
#ax.get_yaxis().tick_left()
#ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

ax2 = ax.twiny()
md = list(map(lambda n: numpy.round(numpy.median(n),2), [cs[0], rnd[0], cs[1], rnd[1], cs[2], rnd[2], cs[3], rnd[3], full[0] ]))
ax2.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), md), weight="bold", fontsize=12)
ax2.set_xticks(list(range(1,10)))
ax2.set_xlim([0, 10])


#legend
hCS, = ax.plot([0, 0], '-', color=blue)
hRND, = ax.plot([0, 0], '-', color=red)
hFULL, = ax.plot([0, 0], '-', color=black)

leg = plt.legend((hCS, hRND, hFULL), ('CDN', 'Uniform', 'Full'), prop={'size': 14}, bbox_to_anchor=(1.0,0.94), loc="upper right")
hCS.set_visible(False)
hRND.set_visible(False)
hFULL.set_visible(False)
for line in leg.get_lines():
    line.set_linewidth(6)

#plt.tight_layout()

plt.savefig("mnist_LL.pdf")
plt.clf()


####################### MNIST

cs = [
      loadtimes("data/MNIST/cs/10"),
      loadtimes("data/MNIST/cs/20"),
      loadtimes("data/MNIST/cs/30"),
      loadtimes("data/MNIST/cs/40"),
      ]

rnd = [
       loadtimes("data/MNIST_good/rnd/10"),
       loadtimes("data/MNIST_good/rnd/20"),
       loadtimes("data/MNIST_good/rnd/30"),
       loadtimes("data/MNIST_good/rnd/40"),
       ]

full = [loadtimes("data/MNIST/full")
        ]

cs = list(map(lambda t: numpy.log(t/3600.0),cs))
rnd = list(map(lambda t: numpy.log(t/3600.0),rnd))
full = list(map(lambda t: numpy.log(t/3600.0),full))

plt.rc('grid', linestyle="--")
plt.rcParams["font.weight"] = "bold"

#fig = plt.figure(1, figsize=(6, 4))


plt.subplots_adjust(left=0.11, bottom=0.13, right=0.999, top=0.90, wspace=0, hspace=0)

# Create an axes instance
ax = fig.add_subplot(111)


# Create the boxplot
boxplot(ax, cs, positions=list(range(1,len(cs)*2,2)), color=blue)

boxplot(ax, rnd, positions=list(range(2,len(rnd)*2+1,2)), color=red)

boxplot(ax, full, positions=[len(cs)+len(rnd)+1], color=black)

ax.set_xlim([0, 10])
ax.set_ylim([1.5, 4.5])
ax.yaxis.grid()


ax.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (s), ['10\%', '20\%', '30\%', '40\%', "100\%"]), weight="bold", fontsize=12)
ax.set_xticks([1.5, 3.5, 5.5, 7.5, 9])

ax.set_yticks(numpy.arange(1.5,4.6,0.5))
ax.set_yticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), numpy.arange(1.0,4.6,0.5)), weight="bold", fontsize=12)

#ax.yaxis.set_tick_params(weight="bold")

plt.axvline(x=4.5, color=blue)
plt.axvline(x=6.5, color=red)

plt.ylabel(r'\textbf{Log Time (in hours)}',fontweight='bold')
plt.xlabel(r'\textbf{Training data (Sample size in percentage)}',fontweight='bold')


ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax2 = ax.twiny()
md = list(map(lambda n: numpy.round(numpy.median(n),2), [cs[0], rnd[0], cs[1], rnd[1], cs[2], rnd[2], cs[3], rnd[3], full[0] ]))
ax2.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), md), weight="bold", fontsize=12)
ax2.set_xticks(list(range(1,10)))
ax2.set_xlim([0, 10])


#legend
hCS, = ax.plot([0, 0], '-', color=blue)
hRND, = ax.plot([0, 0], '-', color=red)
hFULL, = ax.plot([0, 0], '-', color=black)

leg = plt.legend((hCS, hRND, hFULL), ('CDN', 'Uniform', 'Full'), prop={'size': 14}, bbox_to_anchor=(0.0,0.95), loc="upper left")
hCS.set_visible(False)
hRND.set_visible(False)
hFULL.set_visible(False)
for line in leg.get_lines():
    line.set_linewidth(6)

#plt.tight_layout()

plt.savefig("mnist_time.pdf")
plt.clf()


##############################################################


cs = [
      mse("data/MNIST/cs/10", clipped=True, floored=False),
      mse("data/MNIST/cs/20", clipped=True, floored=False),
      mse("data/MNIST/cs/30", clipped=True, floored=False),
      mse("data/MNIST/cs/40", clipped=True, floored=False),
      ]

rnd = [
       mse("data/MNIST_good/rnd/10", clipped=True, floored=False),
       mse("data/MNIST_good/rnd/20", clipped=True, floored=False),
       mse("data/MNIST_good/rnd/30", clipped=True, floored=False),
       mse("data/MNIST_good/rnd/40", clipped=True, floored=False),
       ]

full = [mse("data/MNIST/full", clipped=True, floored=False)
        ]


cs = clog(cs)
rnd = clog(rnd)
full = clog(full)

#fig = plt.figure(1, figsize=(6, 4))

# Create an axes instance
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.12, bottom=0.13, right=0.999, top=0.90, wspace=0, hspace=0)


# Create the boxplot
boxplot(ax, cs, positions=list(range(1,len(cs)*2,2)), color=blue)

boxplot(ax, rnd, positions=list(range(2,len(rnd)*2+1,2)), color=red)

boxplot(ax, full, positions=[len(cs)+len(rnd)+1], color=black)

ax.set_xlim([0, 10])
ax.set_ylim([-2.7, -2.52])
ax.yaxis.grid()

ax.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (s), ['10\%', '20\%', '30\%', '40\%', "100\%"]), weight="bold", fontsize=12)
ax.set_xticks([1.5, 3.5, 5.5, 7.5, 9])

ax.set_yticks(numpy.arange(-2.7,-2.50,0.05))
ax.set_yticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), numpy.arange(-2.7,-2.3,0.05)), weight="bold", fontsize=12)


plt.axvline(x=4.5, color=blue)
plt.axvline(x=6.5, color=red)

#ax.yaxis.set_tick_params(weight="bold")

plt.ylabel(r'\textbf{Log RMSE (Root Mean Square Error)}',fontweight='bold')
plt.xlabel(r'\textbf{Training data (Sample size in percentage)}',fontweight='bold')


ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax2 = ax.twiny()
md = list(map(lambda n: numpy.round(numpy.median(n),2), [cs[0], rnd[0], cs[1], rnd[1], cs[2], rnd[2], cs[3], rnd[3], full[0] ]))
ax2.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), md), weight="bold", fontsize=12)
ax2.set_xticks(list(range(1,10)))
ax2.set_xlim([0, 10])

#legend
hCS, = ax.plot([0, 0], '-', color=blue)
hRND, = ax.plot([0, 0], '-', color=red)
hFULL, = ax.plot([0, 0], '-', color=black)

leg = plt.legend((hCS, hRND, hFULL), ('CDN', 'Uniform', 'Full'), prop={'size': 14}, bbox_to_anchor=(0.99,0.95), loc="upper right")
hCS.set_visible(False)
hRND.set_visible(False)
hFULL.set_visible(False)
for line in leg.get_lines():
    line.set_linewidth(6)

#plt.tight_layout()

plt.savefig("mnist.pdf")
plt.clf()

####################### traffic

cs = [
      loadLL("data/TrafficLarge/cs/10"),
      loadLL("data/TrafficLarge/cs/20"),
      loadLL("data/TrafficLarge/cs/30"),
      loadLL("data/TrafficLarge/cs/40"),
      ]

rnd = [
       loadLL("data/TrafficLarge/rnd/10"),
       loadLL("data/TrafficLarge/rnd/20"),
       loadLL("data/TrafficLarge/rnd/30"),
       loadLL("data/TrafficLarge/rnd/40"),
       ]

full = [loadLL("data/TrafficLarge/full")
        ]


print("empirical eps traffic")
for i in range(len(cs)):
    eepscs = []
    eepsrnd = []
    for fold in range(10):
        eepscs.append(numpy.abs(cs[i][fold] - full[0][fold]) / full[0][fold])
        eepsrnd.append(numpy.abs(rnd[i][fold] - full[0][fold]) / full[0][fold])
    print(numpy.round(numpy.mean(eepscs)*100,2), numpy.round(numpy.mean(eepsrnd)*100,2))



cs = list(map(lambda t: numpy.array(t) / 1000000,cs))
rnd = list(map(lambda t: numpy.array(t) / 1000000,rnd))
full = list(map(lambda t: numpy.array(t) / 1000000,full))


# Create an axes instance
ax = fig.add_subplot(111)
#plt.subplots_adjust(left=0.14, bottom=0.13, right=0.999, top=0.90, wspace=0, hspace=0)
plt.subplots_adjust(left=0.18, bottom=0.13, right=0.999, top=0.90, wspace=0, hspace=0)



# Create the boxplot
boxplot(ax, cs, positions=list(range(1,len(cs)*2,2)), color=blue)

boxplot(ax, rnd, positions=list(range(2,len(rnd)*2+1,2)), color=red)

boxplot(ax, full, positions=[len(cs)+len(rnd)+1], color=black)

ax.set_xlim([0, 10])
ax.set_ylim([3.2, 3.65])
ax.yaxis.grid()

ax.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (s), ['10\%', '20\%', '30\%', '40\%', "100\%"]), weight="bold", fontsize=12)
ax.set_xticks([1.5, 3.5, 5.5, 7.5, 9])

ax.set_yticklabels(map(lambda s: r"$\bm{%s\!\!\times\!\!10^6}$" % (s), ['3.2', '3.3', '3.4', '3.5', "3.6"]), weight="bold", fontsize=12)
ax.set_yticks([3.2, 3.3, 3.4, 3.5, 3.6])

#ax.yaxis.set_tick_params(weight="bold")

plt.axvline(x=2.5, color=blue)
plt.axvline(x=6.5, color=red)

plt.ylabel(r'\textbf{Negative Poisson Pseudo Log--Likelihood}',fontweight='bold')
plt.xlabel(r'\textbf{Training data (Sample size in percentage)}',fontweight='bold')


ax.get_xaxis().tick_bottom()
#ax.get_yaxis().tick_left()
#ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

ax2 = ax.twiny()
md = list(map(lambda n: numpy.round(numpy.median(n),2), [cs[0], rnd[0], cs[1], rnd[1], cs[2], rnd[2], cs[3], rnd[3], full[0] ]))
ax2.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), md), weight="bold", fontsize=12)
ax2.set_xticks(list(range(1,10)))
ax2.set_xlim([0, 10])

#legend
hCS, = ax.plot([0, 0], '-', color=blue)
hRND, = ax.plot([0, 0], '-', color=red)
hFULL, = ax.plot([0, 0], '-', color=black)

leg = plt.legend((hCS, hRND, hFULL), ('CDN', 'Uniform', 'Full'), prop={'size': 14}, bbox_to_anchor=(1.0,0.94), loc="upper right")
hCS.set_visible(False)
hRND.set_visible(False)
hFULL.set_visible(False)
for line in leg.get_lines():
    line.set_linewidth(6)

#plt.tight_layout()

plt.savefig("TrafficLarge_LL.pdf")
plt.clf()


#########################
cs = [
      loadtimes("data/TrafficLarge/cs/10"),
      loadtimes("data/TrafficLarge/cs/20"),
      loadtimes("data/TrafficLarge/cs/30"),
      loadtimes("data/TrafficLarge/cs/40"),
      ]

rnd = [
       loadtimes("data/TrafficLarge/rnd/10"),
       loadtimes("data/TrafficLarge/rnd/20"),
       loadtimes("data/TrafficLarge/rnd/30"),
       loadtimes("data/TrafficLarge/rnd/40"),
       ]

full = [loadtimes("data/TrafficLarge/full")
        ]

cs = list(map(lambda t: numpy.log(t/60.0),cs))
rnd = list(map(lambda t: numpy.log(t/60.0),rnd))
full = list(map(lambda t: numpy.log(t/60.0),full))

plt.rc('grid', linestyle="--")
plt.rcParams["font.weight"] = "bold"

fig = plt.figure(1, figsize=(6.4, 3.5))

# Create an axes instance
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.11, bottom=0.13, right=0.999, top=0.90, wspace=0, hspace=0)



textpos = 2.85
# Create the boxplot
boxplot(ax, cs, positions=list(range(1,len(cs)*2,2)), color=blue)

boxplot(ax, rnd, positions=list(range(2,len(rnd)*2+1,2)), color=red)

boxplot(ax, full, positions=[len(cs)+len(rnd)+1], color=black)

ax.set_xlim([0, 10])
ax.set_ylim([-0.5, 3.0])
ax.yaxis.grid()

ax.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (s), ['10\%', '20\%', '30\%', '40\%', "100\%"]), weight="bold", fontsize=12)
ax.set_xticks([1.5, 3.5, 5.5, 7.5, 9])

ax.set_yticks(numpy.arange(-0.5,3.1,0.5))
ax.set_yticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), numpy.arange(-0.5,3.1,0.5)), weight="bold", fontsize=12)

#ax.yaxis.set_tick_params(weight="bold")

plt.axvline(x=2.5, color=blue)
plt.axvline(x=6.5, color=red)

plt.ylabel(r'\textbf{Log Time (in minutes)}',fontweight='bold')
plt.xlabel(r'\textbf{Training data (Sample size in percentage)}',fontweight='bold')


ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax2 = ax.twiny()
md = list(map(lambda n: numpy.round(numpy.median(n),2), [cs[0], rnd[0], cs[1], rnd[1], cs[2], rnd[2], cs[3], rnd[3], full[0] ]))
ax2.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), md), weight="bold", fontsize=12)
ax2.set_xticks(list(range(1,10)))
ax2.set_xlim([0, 10])


#legend
hCS, = ax.plot([0, 0], '-', color=blue)
hRND, = ax.plot([0, 0], '-', color=red)
hFULL, = ax.plot([0, 0], '-', color=black)

leg = plt.legend((hCS, hRND, hFULL), ('CDN', 'Uniform', 'Full'), prop={'size': 14}, bbox_to_anchor=(0.0,0.95), loc="upper left")
hCS.set_visible(False)
hRND.set_visible(False)
hFULL.set_visible(False)
for line in leg.get_lines():
    line.set_linewidth(6)

#plt.tight_layout()

plt.savefig("TrafficLarge_time.pdf")
plt.clf()



##############################################################


cs = [
      mse("data/TrafficLarge/cs/10", clipped=False, floored=True),
      mse("data/TrafficLarge/cs/20", clipped=False, floored=True),
      mse("data/TrafficLarge/cs/30", clipped=False, floored=True),
      mse("data/TrafficLarge/cs/40", clipped=False, floored=True),
      ]

rnd = [
       mse("data/TrafficLarge/rnd/10", clipped=False, floored=True),
       mse("data/TrafficLarge/rnd/20", clipped=False, floored=True),
       mse("data/TrafficLarge/rnd/30", clipped=False, floored=True),
       mse("data/TrafficLarge/rnd/40", clipped=False, floored=True),
       ]

full = [mse("data/TrafficLarge/full", clipped=False, floored=True),
        ]


cs = clog(cs)
rnd = clog(rnd)
full = clog(full)

#fig = plt.figure(1, figsize=(6, 40))

# Create an axes instance
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.12, bottom=0.13, right=0.999, top=0.90, wspace=0, hspace=0)


boxplot(ax, cs, positions=list(range(1,len(cs)*2,2)), color=blue)

boxplot(ax, rnd, positions=list(range(2,len(rnd)*2+1,2)), color=red)

boxplot(ax, full, positions=[len(cs)+len(rnd)+1], color=black)

ax.set_xlim([0, 10])
ax.set_ylim([1, 2.2])
ax.yaxis.grid()

ax.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (s), ['10\%', '20\%', '30\%', '40\%', "100\%"]), weight="bold", fontsize=12)
ax.set_xticks([1.5, 3.5, 5.5, 7.5, 9])

ax.set_yticks(numpy.arange(1.0,2.3,0.2))
ax.set_yticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), numpy.arange(1.0,2.3,0.2)), weight="bold", fontsize=12)


plt.axvline(x=2.5, color=blue)
plt.axvline(x=6.5, color=red)

#ax.yaxis.set_tick_params(weight="bold")

plt.ylabel(r'\textbf{Log RMSE (Root Mean Square Error)}',fontweight='bold')
plt.xlabel(r'\textbf{Training data (Sample size in percentage)}',fontweight='bold')


ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax2 = ax.twiny()
md = list(map(lambda n: numpy.round(numpy.median(n),2), [cs[0], rnd[0], cs[1], rnd[1], cs[2], rnd[2], cs[3], rnd[3], full[0] ]))
ax2.set_xticklabels(map(lambda s: r"$\bm{%s}$" % (str(s)), md), weight="bold", fontsize=12)
ax2.set_xticks(list(range(1,10)))
ax2.set_xlim([0, 10])


#legend
hCS, = ax.plot([0, 0], '-', color=blue)
hRND, = ax.plot([0, 0], '-', color=red)
hFULL, = ax.plot([0, 0], '-', color=black)

leg = plt.legend((hCS, hRND, hFULL), ('CDN', 'Uniform', 'Full'), prop={'size': 14}, bbox_to_anchor=(0.99,0.95), loc="upper right")
hCS.set_visible(False)
hRND.set_visible(False)
hFULL.set_visible(False)
for line in leg.get_lines():
    line.set_linewidth(6)

#plt.tight_layout()

plt.savefig("TrafficLarge.pdf")
plt.clf()




