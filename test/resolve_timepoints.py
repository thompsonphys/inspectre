import math, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/sf/inspectre"))
from inspectre.inspectre import Inspectre

SPIN,P,ECC = 0.9,10.0,0.2
PTS = (("near d=5.7e-04", 8.33859, 6.780454e-05, 5.654e-4),
       ("d=3e-02       ", 8.68029, -3.455536e-03, 3.0e-2),
       ("C1 max        ", 19.96312, -4.209789e-01, 11.44))
MS=(2,6); NS=(2,8,16)
NMAX=65536
SUBS=(256,512,1024,2048,4096,8192,16384,32768,65536)

insp=Inspectre(spin=SPIN, semilatus_rectum=P, eccentricity=ECC)
Tr=insp.t_from_lambda(insp.mino_period_r); om_r,om_ph=insp.omega_r,insp.omega_phi
ts=np.arange(NMAX)*(Tr/NMAX)
print(f"Tr={Tr:.5f}  NMAX={NMAX}   n_samp = 1.374*N_t*d\n")

for tag,r_f,cos_th,d in PTS:
    th_f=math.acos(cos_th)
    ser={m: np.empty(NMAX,dtype=complex) for m in MS}
    for i,t in enumerate(ts):
        lam=insp.lambda_from_t(t)
        insp.set_particle(insp.r_from_lambda(lam), math.pi/2,
                          insp.phi_from_lambda(lam), insp.four_velocity_equatorial(lam))
        for m in MS:
            o=insp.es.calc_m(m,r_f,th_f)
            ser[m][i]=complex(o["src"][0],o["src"][1])
    print(f"=== {tag} d={d:.3e}   n_samp: " +
          " ".join(f"N{N}:{1.374*N*d:.1f}" for N in (512,8192,65536)) + " ===")
    for m in MS:
        def mode(N,n):
            sub=ser[m][::NMAX//N]; tt=ts[::NMAX//N]
            return np.sum(sub*np.exp(1j*(m*om_ph+n*om_r)*tt))/N
        for n in NS:
            ref=mode(NMAX,n)
            vals=[abs(mode(N,n)-ref)/abs(ref) for N in SUBS[:-1]]
            print(f"  m={m} n={n:<2d} |src_mn|={abs(ref):.3e}  " +
                  " ".join(f"{N//1024 if N>=1024 else N}{'k' if N>=1024 else ''}:{v:.1e}"
                           for N,v in zip(SUBS[:-1],vals)))
    print(flush=True)
