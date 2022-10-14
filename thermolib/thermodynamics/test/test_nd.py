from thermolib.thermodynamics.histogram import HistogramND,Histogram2D,Histogram1D
import itertools
import numpy as np

def test_ND():
    for ncv in range(1,4):

        num_bin_points = [np.random.randint(5,10) for _ in range(ncv)]
        r = [ np.random.rand()*10 for _ in range(ncv)]
        bins = [np.linspace(start= 0.0, stop =r[i],num=num_bin_points[i])  for i in range(ncv)]

        class Bias:
            def __init__(self,q0) -> None:
                self.q0 = q0

            def __call__(self, *q):
                out = 0.5*( np.sum([  (qi-q0i)**2 for qi,q0i in zip(q,self.q0) ] , axis=0 ))
                return out

        trajs = []
        biasses = [     ]

        traj_length = 200
        for qn in itertools.product( *bins ):
            b = Bias(  qn ) 

            #generate boltzman samples
            samples = np.random.rand( 100000,ncv) *r
            biases = b( *[samples[:,i] for i in range(ncv)] )
            p = np.exp( -biases )
            p /= np.sum(p)
            ind = np.random.choice(
                a = np.arange(100000  ),
                size = traj_length,
                p =  p 
            )

            biasses.append( b  )
            trajs.append(  samples[ind,:]  )

        h1 = HistogramND.from_wham(bins=bins, temp=1,trajectories=trajs,biasses=biasses,error_estimate="mle_f",)

        h0 = None
        if ncv == 1:
             h0 = Histogram1D.from_wham( bins=bins[0], temp=1 ,trajectories=trajs,biasses=biasses,error_estimate="mle_f",)
        elif ncv ==2:
            h0 = Histogram2D.from_wham( bins=bins, temp=1 ,trajectories=trajs,biasses=biasses,error_estimate="mle_f",)

        if h0 is not None:
            assert ((h0.ps-h1.ps) < 1e-12).all()
            assert ((h0.pupper-h1.pupper) < 1e-12).all()
            assert ((h0.plower-h1.plower) < 1e-12).all()


if __name__=='__main__':
    test_ND()