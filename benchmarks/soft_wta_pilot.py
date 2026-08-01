"""Pilot: does soft-WTA make the rule data-sensitive, where hard-WTA is flat?"""
import numpy as np, sys
sys.path.insert(0,"benchmarks")
from pathways import FRAME, cluster_auc, place
from real_binding import split
from neurobrain.sensing.natural import load_audiovisual, load_cifar10
from neurobrain.learning.selforganize import develop_v1
from neurobrain.vision.widev1 import PopulationAdaptation, WideV1
from neurobrain.cognition.multimodal import _unit

def develop_soft(layer, images, epochs=3, lr0=0.35, lr1=0.02, temp=0.25, seed=0):
    """SoftHebb-style: a softmax over the column replaces argmax + neighbourhood.

    The theory (Moraitis et al., Neuromorph. Comput. Eng. 2022) is that a SOFT
    winner-take-all network with Hebbian plasticity maintains a Bayesian
    generative model of the data -- it is doing density estimation. Hard WTA is
    vector quantization: k centroids that converge and then stop moving, which
    is exactly a rule that cannot use more data."""
    rng = np.random.default_rng(seed)
    n_pos, nf = layer.n_pos, layer.n_cells // layer.n_pos
    head = nf * n_pos
    W = layer.Wt[:head].reshape(nf, n_pos, -1)
    total = max(1, epochs*len(images)); step = 0
    for _ in range(epochs):
        for i in rng.permutation(len(images)):
            lr = lr0 * (lr1/lr0) ** (step/total); step += 1
            P = layer.patches(images[i])
            e = np.linalg.norm(P, axis=1)
            live = np.flatnonzero(e > 0.15*(e.max()+1e-9))
            if not len(live): continue
            take = live if len(live) <= 24 else rng.choice(live, 24, replace=False)
            Pn = P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-6)
            for p in take:
                d = W[:, p, :] @ Pn[p]
                q = np.exp((d - d.max())/temp); q /= q.sum()      # soft-WTA
                W[:, p, :] += (lr*q)[:,None] * (Pn[p][None,:] - W[:, p, :])
            W[:] = W.mean(axis=1, keepdims=True)                  # tie
            np.maximum(W,0.0,out=W)
            W /= np.maximum(np.linalg.norm(W,axis=2,keepdims=True),1e-6)
    layer.Wt[:head] = W.reshape(head,-1)
    return layer

imgs,_w,y,names = load_audiovisual(n_per_class=40, seed=0, grayscale=True, size=32)
y=np.asarray(y,int); frames=[place(im) for im in imgs]
trX,trY,teX,teY = load_cifar10(n_train=50000,n_test=10000,seed=0,size=32,grayscale=True)
X=np.concatenate([trX,teX]); Y=np.concatenate([trY,teY])
CID={"airplane":0,"automobile":1,"bird":2,"cat":3,"dog":5,"frog":6}
want=[CID[c] for c in names]

print("PILOT: is the flatness caused by HARD winner-take-all?\n")
print(f"{'rule':<12}{'240 img':>10}{'2400 img':>10}{'12000 img':>11}{'slope':>9}")
for tag, fn in (("hard-WTA", None), ("soft-WTA", develop_soft)):
    got=[]
    for n_per in (40, 400, 2000):
        pool=np.concatenate([np.flatnonzero(Y==c)[:n_per] for c in want])
        devel=[place(X[i]) for i in pool]
        per=[]
        for sd in (0,1,2):
            v=WideV1(n_cells=4096, window_ms=50, rf=7, stride=2,
                     image_shape=(FRAME,FRAME), seed=sd)
            if fn is None: develop_v1(v, devel, epochs=3, tie=True, seed=sd)
            else: fn(v, devel, epochs=3, seed=sd)
            R=np.array([v.drive(f) for f in frames],np.float32)
            ad=PopulationAdaptation(R.shape[1]); V=np.array([_unit(ad(r)) for r in R],np.float32)
            tr,te=split(y,sd)
            per.append(cluster_auc(V[te],y[te],V[tr],y[tr],np.random.default_rng(sd)))
        got.append(float(np.mean(per)))
    print(f"{tag:<12}{got[0]:>10.4f}{got[1]:>10.4f}{got[2]:>11.4f}{got[2]-got[0]:>+9.4f}", flush=True)
print("\n  reference: CNN gains +0.123 over the same range; hard-WTA gained -0.013")
