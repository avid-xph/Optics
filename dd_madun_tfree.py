"""DD-MADUN T-free: sequence-length-agnostic DD-MADUN (based on dd_madun_v3.py)

Differences from DD_MADUNv3:
  - DCT matrix is built on the fly from the actual T of the input (cached per
    (T, device, dtype)) instead of a fixed T x T buffer.
  - Denoiser no longer uses T as its channel count. Frames are folded into the
    batch and processed by shared 2D convs (per-frame weight sharing), with
    depthwise temporal convs (k=3 along T) inside every residual block and a
    sequence-mean context channel at the head. Same weights work for any T.
  - Everything else (momentum DC, ADMM-style split, soft-threshold stages, GRU
    state across stages, adaptive noise map, mask_w / Plan B) follows v3, and
    forward(E, fwd, adj, mask_w) is called exactly like v3 with CUPForwardModel.

Mixed-T training entry (this file):
  python dd_madun_tfree.py [--batch_size 2] [--epochs 150] [--T_list 8,12,16,20,24,28,32]
  Every batch draws a random T from T_list; clips are stored at T_max and a
  random temporal window of length T is cropped from each.
"""
import torch,torch.nn as nn,torch.nn.functional as F,numpy as np
from torch.utils.checkpoint import checkpoint
from dd_madun_v3 import (build_dct_matrix,dct_along_time,idct_along_time,CBAMBlock,
                         DataConsistencyMomentum,freq_band_l1,edge_loss,ssim_loss,
                         FocalFrequencyLoss,VGGPerceptualLoss)

_DCT_CACHE={}
def get_dct_matrix(T,device=None,dtype=torch.float32):
  """T x T orthonormal DCT-II matrix, cached per (T, device, dtype)."""
  key=(int(T),str(device),dtype)
  dm=_DCT_CACHE.get(key)
  if dm is None:
    dm=build_dct_matrix(int(T)).to(device=device,dtype=dtype)
    _DCT_CACHE[key]=dm
  return dm

class TemporalMix(nn.Module):
  """Depthwise conv along T on per-frame features (B*T,C,H,W). Zero-init -> identity at start."""
  def __init__(self,nf,k=3):
    super().__init__()
    self.conv=nn.Conv3d(nf,nf,(k,1,1),padding=(k//2,0,0),groups=nf,bias=False)
    nn.init.zeros_(self.conv.weight)
  def forward(self,f,T):
    BT,C,H,W=f.shape;B=BT//T
    f5=f.view(B,T,C,H,W).permute(0,2,1,3,4)
    d=self.conv(f5).permute(0,2,1,3,4).reshape(BT,C,H,W)
    return f+d

class STCBAMResBlock(nn.Module):
  """CBAMResBlock (v3) + temporal mixing; weights shared over frames."""
  def __init__(self,nf,r=8):
    super().__init__()
    self.c1=nn.Conv2d(nf,nf,3,padding=1,bias=False)
    self.c2=nn.Conv2d(nf,nf,3,padding=1,bias=False)
    self.relu=nn.ReLU(True);self.cbam=CBAMBlock(nf,r);self.tm=TemporalMix(nf)
  def forward(self,x,T):
    r=x;x=self.relu(self.c1(x));x=self.c2(x)
    x=self.cbam(x);x=self.tm(x,T);return self.relu(x+r)

class FFDNetGRUTFree(nn.Module):
  """T-agnostic version of FFDNetGRUv3.
  Input per frame: [frame, noise map, sequence mean] -> 3 channels, independent of T.
  GRU hidden state is per frame: (B*T, nf, H, W)."""
  def __init__(self,nf=48,nb=6,use_ckpt=False):
    super().__init__()
    self.use_ckpt=use_ckpt
    self.head=nn.Sequential(nn.Conv2d(3,nf,3,1,1,bias=False),nn.ReLU(True))
    self.gz=nn.Conv2d(nf*2,nf,3,1,1,bias=True)
    self.gr=nn.Conv2d(nf*2,nf,3,1,1,bias=True)
    self.gh=nn.Conv2d(nf*2,nf,3,1,1,bias=True)
    self.blocks=nn.ModuleList([STCBAMResBlock(nf) for _ in range(nb)])
    self.sf=nn.Conv2d(nf*2,nf,1,bias=False)
    self.tail=nn.Conv2d(nf,1,3,1,1,bias=False)
  def forward(self,x,sig,hp=None):
    B,T,H,W=x.shape
    if sig.dim()<=2:sig=sig.view(-1,1,1,1).expand(B,1,H,W)
    sig=sig.to(x.device)
    sg=sig.unsqueeze(1).expand(B,T,1,H,W)
    xm=x.mean(1,keepdim=True).unsqueeze(2).expand(B,T,1,H,W)
    inp=torch.cat([x.unsqueeze(2),sg,xm],dim=2).reshape(B*T,3,H,W)
    hf=self.head(inp)
    if hp is None or hp.shape[0]!=B*T:hp=torch.zeros_like(hf)
    gi=torch.cat([hf,hp],dim=1)
    zg=torch.sigmoid(self.gz(gi))
    rg=torch.sigmoid(self.gr(gi))
    hc=torch.tanh(self.gh(torch.cat([hf,rg*hp],dim=1)))
    h=(1-zg)*hp+zg*hc
    feat=h
    for b in self.blocks:
      if self.use_ckpt and feat.requires_grad:
        feat=checkpoint(b,feat,T,use_reentrant=False)
      else:
        feat=b(feat,T)
    fused=self.sf(torch.cat([hf,feat],dim=1))
    return x-self.tail(fused).view(B,T,H,W),h

class DD_MADUN_TFree(nn.Module):
  """Same interface as DD_MADUNv3: forward(E, fwd, adj, mask_w=None) -> (x, inter).
  T is taken from adj(E) at run time; the T argument is accepted only for
  constructor compatibility and is ignored."""
  def __init__(self,T=None,H=None,W=None,K=8,K1=3,ng=6,nf=48,nb=6,use_ckpt=False):
    super().__init__()
    self.H=H;self.W=W;self.K=K;self.K1=K1;self.K2=K-K1
    self.lrs=nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(K)])
    self.lrf=nn.ParameterList([nn.Parameter(torch.tensor(-0.5)) for _ in range(K)])
    self.llf=nn.ParameterList([nn.Parameter(torch.tensor(-0.5)) for _ in range(K)])
    self.lmu=nn.ParameterList([nn.Parameter(torch.tensor(2.0)) for _ in range(K)])
    self.let=nn.ParameterList([nn.Parameter(torch.tensor(-1.5)) for _ in range(K)])
    self.lsg=nn.ParameterList([nn.Parameter(torch.tensor(-1.0)) for _ in range(self.K2)])
    self.lbt=nn.ParameterList([nn.Parameter(torch.tensor(-1.0)) for _ in range(self.K2)])
    self.dc=nn.ModuleList([DataConsistencyMomentum(ng) for _ in range(K)])
    self.den=nn.ModuleList([FFDNetGRUTFree(nf,nb,use_ckpt) for _ in range(self.K2)])
    self._iw()
  def _iw(self):
    # Conv3d (TemporalMix) keeps its zero init
    for m in self.modules():
      if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        if m.bias is not None:nn.init.zeros_(m.bias)
  def get_dm(self,T,device=None,dtype=torch.float32):
    return get_dct_matrix(T,device,dtype)
  def _anm(self,E,sk,bk,W_rec=None):
    """adaptive noise map from measurement E; crop to reconstruction width W_rec."""
    B,H,Wm=E.shape
    gx=torch.abs(E-torch.roll(E,1,-1));gx[:,:,0]=gx[:,:,1]
    gm=gx.amin((1,2),True);gx2=gx.amax((1,2),True)
    gn=torch.where(gx2>gm+1e-8,(gx-gm)/(gx2-gm),torch.zeros_like(gx))
    nm=(sk*(1.0+bk*gn)).unsqueeze(1)        # (B,1,H,Wm)
    Wc = W_rec if W_rec is not None else self.W
    return nm[:,:,:,:Wc]                      # (B,1,H,W_rec)
  def forward(self,E,fwd,adj,mask_w=None):
    x=adj(E)
    dm=self.get_dm(x.shape[1],x.device,x.dtype)
    if mask_w is not None:
      xb=x*mask_w
      msum=mask_w.sum(dim=[2,3],keepdim=True).clamp(min=1e-6)
      mmean=xb.sum(dim=[2,3],keepdim=True)/msum   # (B,1,1,1)
      x=mask_w*x+(1.0-mask_w)*mmean
    zs=x.clone();us=torch.zeros_like(x)
    zf=dct_along_time(x,dm);uf=torch.zeros_like(zf)
    v=torch.zeros_like(x);h=None;inter=[]
    for k in range(self.K):
      rs=torch.exp(self.lrs[k]);rf=torch.exp(self.lrf[k])
      lf=torch.exp(self.llf[k]);mk=torch.sigmoid(self.lmu[k]);ek=torch.exp(self.let[k])
      x,v=self.dc[k](x,v,zs,us,zf,uf,E,fwd,adj,dm,rs,rf,mk,ek,mask_w)
      vs=x+us
      if k<self.K1:
        ts=1.0/(rs+1e-8)
        zs=torch.sign(vs)*torch.relu(torch.abs(vs)-ts)
      else:
        idx=k-self.K1;sk=torch.exp(self.lsg[idx]);bk=torch.exp(self.lbt[idx])
        nm=self._anm(E,sk,bk,W_rec=x.shape[-1]);zs,h=self.den[idx](vs,nm,h)
      xd=dct_along_time(x,dm)
      vf=xd+uf
      tf=rf/(lf+1e-8)
      zf=torch.sign(vf)*torch.relu(torch.abs(vf)-tf)
      us=us+x-zs;uf=uf+xd-zf
      inter.append(x.clone())
    return x,inter
  def get_p(self):
    return {'rs':[torch.exp(p).item() for p in self.lrs],
            'rf':[torch.exp(p).item() for p in self.lrf],
            'lf':[torch.exp(p).item() for p in self.llf],
            'mu':[torch.sigmoid(p).item() for p in self.lmu],
            'et':[torch.exp(p).item() for p in self.let],
            'sg':[torch.exp(p).item() for p in self.lsg],
            'bt':[torch.exp(p).item() for p in self.lbt]}

class DDMADUNLossTFree(nn.Module):
  """DDMADUNLossv3 with the frequency-band term using a DCT matrix of the actual T."""
  def __init__(self, as_=0.15, gf=0.15, sw=0.20, ew=0.05,
               ffl_w=0.0, vgg_w=0.0, device="cuda"):
    super().__init__()
    self.a=as_; self.g=gf; self.sw=sw; self.ew=ew
    self.ffl_w=ffl_w; self.vgg_w=vgg_w
    self.ffl_loss = FocalFrequencyLoss(alpha=1.0) if ffl_w > 0 else None
    self.vgg_loss = VGGPerceptualLoss(device=device) if vgg_w > 0 else None
  def forward(self, Ip, Ig, inter=None):
    l1=F.l1_loss(Ip,Ig); tl=l1; ld={"l1":l1.item()}
    if inter is not None:
      K=len(inter); ls=0.0
      for k,xk in enumerate(inter): ls += ((k+1)/K)*F.l1_loss(xk,Ig)
      ls/=K; tl+=self.a*ls; ld["st"]=ls.item()
    if self.g>0:
      T=Ip.shape[1]
      lf=freq_band_l1(Ip,Ig,get_dct_matrix(T,Ip.device,Ip.dtype),T=T); tl+=self.g*lf; ld["fb"]=lf.item()
    if self.sw>0:
      lss=ssim_loss(Ip,Ig); tl+=self.sw*lss; ld["ss"]=lss.item()
    if self.ew>0:
      le=edge_loss(Ip,Ig); tl+=self.ew*le; ld["ed"]=le.item()
    if self.ffl_loss is not None and self.ffl_w>0:
      lffl=self.ffl_loss(Ip,Ig); tl+=self.ffl_w*lffl; ld["ffl"]=lffl.item()
    if self.vgg_loss is not None and self.vgg_w>0:
      lvgg=self.vgg_loss(Ip,Ig); tl+=self.vgg_w*lvgg; ld["vgg"]=lvgg.item()
    ld["total"]=tl.item()
    return tl, ld


# =====================================================================
#  Mixed-T training entry
# =====================================================================
def create_mixed_dataset_tmax(n_train, n_test, T_max, H, W, real_ratio, davis_root,
                              seed=42, n_objects=7, store_dtype=torch.float16, chunk=100):
  """Like video_dataset.create_mixed_dataset, but clips are T_max long and stored
  in store_dtype (fp16 by default) to keep RAM manageable. Only DAVIS sequences
  with >= T_max frames are used."""
  from video_dataset import (_load_davis_sequences,_load_clip,_augment_clip,
                             make_realistic_synthetic)
  from dataset import make_complex_scene
  rng=np.random.RandomState(seed)
  seqs=[s for s in _load_davis_sequences(davis_root) if len(s)>=T_max]
  n_real=int(n_train*real_ratio) if len(seqs)>0 else 0
  n_synth=n_train-n_real
  print(f"  [MixedDataset-Tmax] T_max={T_max}, usable DAVIS sequences: {len(seqs)}")
  print(f"  [MixedDataset-Tmax] Train: {n_real} real + {n_synth} synth = {n_train}")
  train=torch.empty(n_train,T_max,H,W,dtype=store_dtype)
  for i in range(n_real):
    clip=_augment_clip(_load_clip(seqs[rng.randint(0,len(seqs))],T_max,H,W))
    train[i]=torch.from_numpy(clip).to(store_dtype)
  i=n_real
  while i<n_train:
    nb=min(chunk,n_train-i)
    s=make_realistic_synthetic(nb,T_max,H,W,seed=rng.randint(0,10000),n_objects=n_objects)
    train[i:i+nb]=s.to(store_dtype);i+=nb
  train=train[torch.randperm(n_train)]
  test=make_complex_scene(n_test,T_max,H,W,seed=rng.randint(10000,20000),n_objects=n_objects)
  return train,test.float()

def _build_exp_cdf(mask_path):
  exp=np.load(mask_path)
  vals=exp.flatten().astype(np.float32)
  hist,edges=np.histogram(vals,bins=256,range=(0,1))
  cdf=np.cumsum(hist)/hist.sum()
  centers=(edges[:-1]+edges[1:])/2
  stats={"mean":float(vals.mean()),"frac_near_dark":float((vals<0.05).mean()),
         "frac_bright":float((vals>0.5).mean())}
  return cdf.astype(np.float32),centers.astype(np.float32),stats

def sample_grayscale_mask(H,W,device,exp_cdf=None,exp_centers=None,blur=2.0,seed=None):
  """Histogram-matched smooth random mask (same as dd_madun_v3_train_p4)."""
  from scipy.ndimage import gaussian_filter
  if seed is not None:
    g=torch.Generator().manual_seed(seed)
    base=torch.bernoulli(torch.full((H,W),0.5),generator=g).float()
  else:
    base=torch.bernoulli(torch.full((H,W),0.5)).float()
  field=gaussian_filter(base.numpy(),sigma=blur)
  field=(field-field.min())/(field.max()-field.min()+1e-8)
  flat=field.flatten();order=np.argsort(flat);n=len(flat)
  matched=np.empty(n,dtype=np.float32)
  matched[order]=np.interp(np.linspace(0,1,n),exp_cdf,exp_centers)
  return torch.from_numpy(matched.reshape(H,W)).to(device)

def make_mask_w(mask,thr,soft):
  M=mask.view(1,1,mask.shape[-2],mask.shape[-1])
  return torch.sigmoid((M-thr)/max(soft,1e-4)).detach()

class MultiTCUP:
  """One CUPForwardModel per T, all sharing the same encoding mask."""
  def __init__(self,H,W,T_list,shear,device):
    from cup_simulator import CUPForwardModel
    self.cups={T:CUPForwardModel(H=H,W=W,T=T,shear_per_frame=shear).to(device) for T in T_list}
    self.set_mask(self.cups[T_list[0]].mask)
  def set_mask(self,mask):
    for c in self.cups.values():c.set_mask(mask)
  @property
  def mask(self):
    return next(iter(self.cups.values())).mask
  def __getitem__(self,T):
    return self.cups[T]

def augment(gt):
  """flip + rotate 90 + brightness jitter (same as P4)."""
  for b in range(gt.shape[0]):
    if np.random.rand()>0.5:gt[b]=torch.flip(gt[b],dims=[1])
    if np.random.rand()>0.5:gt[b]=torch.flip(gt[b],dims=[2])
    if np.random.rand()>0.5 and gt.shape[-1]==gt.shape[-2]:
      gt[b]=torch.rot90(gt[b],k=np.random.randint(1,4),dims=[1,2])
    if np.random.rand()>0.5:
      gt[b]=torch.clamp(gt[b]*np.random.uniform(0.9,1.1)+np.random.uniform(-0.05,0.05),0,1)
  return gt

def crop_time(clips,T,random_start=True):
  """clips: (B,T_max,H,W) -> (B,T,H,W) contiguous temporal windows."""
  Tm=clips.shape[1]
  if T==Tm:return clips
  if not random_start:return clips[:,:T]
  st=np.random.randint(0,Tm-T+1,size=clips.shape[0])
  return torch.stack([clips[i,s:s+T] for i,s in enumerate(st)],0)

def main():
  import os,time,json,copy,math,argparse
  import torch.optim as optim
  from baselines import compute_psnr,compute_ssim

  p=argparse.ArgumentParser(description="DD-MADUN T-free mixed-T training")
  p.add_argument("--T_list",type=str,default="8,12,16,20,24,28,32")
  p.add_argument("--resume",type=str,default=None)
  p.add_argument("--continue_train",action="store_true")
  p.add_argument("--batch_size",type=int,default=2)
  p.add_argument("--frame_budget",type=int,default=0,
                 help="If >0, per-batch size = max(1, frame_budget // T) (keeps memory ~constant over T)")
  p.add_argument("--epochs",type=int,default=150)
  p.add_argument("--lr",type=float,default=1e-3)
  p.add_argument("--n_train",type=int,default=1000)
  p.add_argument("--n_test",type=int,default=10)
  p.add_argument("--H",type=int,default=256)
  p.add_argument("--W",type=int,default=256)
  p.add_argument("--K",type=int,default=8)
  p.add_argument("--K1",type=int,default=3)
  p.add_argument("--n_gd_steps",type=int,default=6)
  p.add_argument("--nf",type=int,default=48)
  p.add_argument("--nb",type=int,default=6)
  p.add_argument("--use_ckpt",action="store_true",help="Gradient checkpointing in denoiser blocks")
  p.add_argument("--real_ratio",type=float,default=0.7)
  p.add_argument("--ffl_weight",type=float,default=0.10)
  p.add_argument("--vgg_weight",type=float,default=0.02)
  p.add_argument("--out_dir",type=str,default="output_tfree")
  p.add_argument("--davis_root",type=str,default="DAVIS")
  p.add_argument("--gray_mask",action="store_true")
  p.add_argument("--gray_blur",type=float,default=2.0)
  p.add_argument("--gray_seed",type=int,default=12345)
  p.add_argument("--mask_dc",action="store_true")
  p.add_argument("--mask_dc_thr",type=float,default=0.05)
  p.add_argument("--mask_dc_soft",type=float,default=0.05)
  args=p.parse_args()

  T_list=sorted(int(t) for t in args.T_list.split(","))
  T_max=max(T_list)
  C={"T_list":T_list,"H":args.H,"W":args.W,"K":args.K,"K1":args.K1,
     "n_gd_steps":args.n_gd_steps,"denoiser_feats":args.nf,"denoiser_blocks":args.nb,
     "n_train":args.n_train,"n_test":args.n_test,"batch_size":args.batch_size,
     "frame_budget":args.frame_budget,"epochs":args.epochs,"lr":args.lr,
     "warmup_epochs":10,"weight_decay":1e-5,"noise_std":0.01,
     "alpha_stage":0.15,"gamma_freq":0.15,"ssim_weight":0.20,"edge_weight":0.05,
     "ffl_weight":args.ffl_weight,"vgg_weight":args.vgg_weight,
     "ema_decay":0.999,"shear_per_frame":1,"real_ratio":args.real_ratio,
     "device":"cuda","output_dir":args.out_dir,"davis_root":args.davis_root,
     "seed":42,"n_objects":7,"resume":args.resume,
     "gray_mask":args.gray_mask,"gray_blur":args.gray_blur,"gray_seed":args.gray_seed,
     "mask_dc":args.mask_dc,"mask_dc_thr":args.mask_dc_thr,"mask_dc_soft":args.mask_dc_soft}
  C["use_ckpt"]=args.use_ckpt
  def cfg_safe():
    return {k:v for k,v in C.items()}
  dev=C["device"]
  os.makedirs(C["output_dir"],exist_ok=True)
  torch.manual_seed(C["seed"]);np.random.seed(C["seed"])

  print("="*70)
  print("  DD-MADUN T-free: mixed-T training")
  print("  T_list=%s, H=%d, W=%d, K=%d, K1=%d" % (T_list,C["H"],C["W"],C["K"],C["K1"]))
  print("  Denoiser: feats=%d, blocks=%d, GD steps=%d, ckpt=%s" % (args.nf,args.nb,args.n_gd_steps,args.use_ckpt))
  print("="*70)

  exp_cdf=exp_centers=None
  if C["gray_mask"]:
    here=os.path.dirname(os.path.abspath(__file__))
    mp=os.path.join(here,"..","实验数据","7-28 10hz 3vpp","encoding_mask_normalized.npy")
    assert os.path.exists(mp),f"Experimental mask not found: {mp}"
    exp_cdf,exp_centers,st=_build_exp_cdf(mp)
    print(f"  [GrayMask] mean={st['mean']:.3f}, near-dark={100*st['frac_near_dark']:.0f}%")

  print("\n[1/5] Creating dataset at T_max=%d..." % T_max)
  train_data,test_data=create_mixed_dataset_tmax(C["n_train"],C["n_test"],T_max,C["H"],C["W"],
      C["real_ratio"],C["davis_root"],seed=C["seed"],n_objects=C["n_objects"])
  print(f"  Train: {tuple(train_data.shape)} {train_data.dtype}, Test: {tuple(test_data.shape)}")

  cup=MultiTCUP(C["H"],C["W"],T_list,C["shear_per_frame"],dev)

  print("\n[2/5] Building DD_MADUN_TFree...")
  model=DD_MADUN_TFree(H=C["H"],W=C["W"],K=C["K"],K1=C["K1"],ng=C["n_gd_steps"],
                       nf=args.nf,nb=args.nb,use_ckpt=args.use_ckpt)
  print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
  criterion=DDMADUNLossTFree(as_=C["alpha_stage"],gf=C["gamma_freq"],sw=C["ssim_weight"],
      ew=C["edge_weight"],ffl_w=C["ffl_weight"],vgg_w=C["vgg_weight"],device=dev)

  optimizer=optim.Adam(model.parameters(),lr=C["lr"],weight_decay=C["weight_decay"])
  def lr_lambda(epoch):
    if epoch<C["warmup_epochs"]:return (epoch+1)/C["warmup_epochs"]
    q=(epoch-C["warmup_epochs"])/max(1,C["epochs"]-C["warmup_epochs"])
    return 0.5*(1+math.cos(math.pi*q))
  scheduler=optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)

  start_epoch=1
  if C["resume"] and os.path.exists(C["resume"]):
    ckpt=torch.load(C["resume"],map_location=dev,weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    if args.continue_train:
      if "optimizer_state_dict" in ckpt:optimizer.load_state_dict(ckpt["optimizer_state_dict"])
      if "epoch" in ckpt:
        start_epoch=ckpt["epoch"]+1
        for _ in range(start_epoch-1):scheduler.step()
    print(f"  Resumed from {C['resume']} (start_epoch={start_epoch})")

  model=model.to(dev)
  ema_model=copy.deepcopy(model)
  if C["resume"] and args.continue_train and os.path.exists(C["resume"]):
    if "ema_model_state_dict" in ckpt:ema_model.load_state_dict(ckpt["ema_model_state_dict"])
  for q in ema_model.parameters():q.requires_grad_(False)
  def update_ema(ema,m,decay):
    with torch.no_grad():
      for ep,mp_ in zip(ema.parameters(),m.parameters()):
        ep.data.mul_(decay).add_(mp_.data,alpha=1-decay)

  def eval_mask():
    if C["gray_mask"]:
      cup.set_mask(sample_grayscale_mask(C["H"],C["W"],dev,exp_cdf,exp_centers,
                                         C["gray_blur"],seed=C["gray_seed"]))
    return make_mask_w(cup.mask,C["mask_dc_thr"],C["mask_dc_soft"]) if C["mask_dc"] else None

  def evaluate(net,mask_w,verbose=False):
    """Per-T PSNR/SSIM on the first T frames of each test clip."""
    res={}
    with torch.no_grad():
      for T in T_list:
        ps,ss=[],[]
        for i in range(C["n_test"]):
          gt=test_data[i:i+1,:T].to(dev)
          E=cup[T].forward(gt);E=E+C["noise_std"]*torch.randn_like(E)
          Ip,_=net(E,cup[T].forward,cup[T].adjoint,mask_w)
          Ip=torch.clamp(Ip,0,1)
          ps.append(float(compute_psnr(Ip,gt)));ss.append(float(compute_ssim(Ip,gt)))
        res[T]=(float(np.mean(ps)),float(np.mean(ss)))
        if verbose:print(f"    T={T:2d}: PSNR={res[T][0]:.2f} SSIM={res[T][1]:.4f}")
    return res

  print(f"\n[3/5] Training on {dev} (epochs {start_epoch}-{C['epochs']})...")
  frames_per_epoch=C["n_train"]
  history={"epoch":[],"loss":[],"lr":[],"psnr":[],"ema_psnr":[],"ema_ssim":[],"per_T":[]}
  best_ema_psnr=0.0;best_epoch=0;t_start=time.time()
  for epoch in range(start_epoch,C["epochs"]+1):
    model.train()
    perm=torch.randperm(C["n_train"]);ptr=0
    epoch_loss=0.0;nbat=0;detail={};t_count={T:0 for T in T_list}
    while ptr<frames_per_epoch:
      T=int(np.random.choice(T_list))
      bs=max(1,C["frame_budget"]//T) if C["frame_budget"]>0 else C["batch_size"]
      idx=perm[ptr:ptr+bs];ptr+=bs
      if len(idx)==0:break
      gt=crop_time(train_data[idx],T).float().to(dev)
      gt=augment(gt)
      if C["gray_mask"]:
        cup.set_mask(sample_grayscale_mask(C["H"],C["W"],dev,exp_cdf,exp_centers,C["gray_blur"]))
      mask_w=make_mask_w(cup.mask,C["mask_dc_thr"],C["mask_dc_soft"]) if C["mask_dc"] else None
      E=cup[T].forward(gt);E=E+C["noise_std"]*torch.randn_like(E)
      I_pred,inter=model(E,cup[T].forward,cup[T].adjoint,mask_w)
      loss,ld=criterion(I_pred,gt,inter)
      optimizer.zero_grad();loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
      optimizer.step();update_ema(ema_model,model,C["ema_decay"])
      epoch_loss+=loss.item();nbat+=1;t_count[T]+=1
      for k,v in ld.items():detail[k]=detail.get(k,0)+v
    scheduler.step()
    avg_loss=epoch_loss/max(1,nbat)

    if epoch==start_epoch or epoch%5==0 or epoch==C["epochs"]:
      model.eval();ema_model.eval()
      mask_w=eval_mask()
      r=evaluate(model,mask_w);re_=evaluate(ema_model,mask_w)
      ap=float(np.mean([v[0] for v in r.values()]))
      ea=float(np.mean([v[0] for v in re_.values()]));eas=float(np.mean([v[1] for v in re_.values()]))
      lr_now=optimizer.param_groups[0]["lr"]
      history["epoch"].append(epoch);history["loss"].append(avg_loss);history["lr"].append(lr_now)
      history["psnr"].append(ap);history["ema_psnr"].append(ea);history["ema_ssim"].append(eas)
      history["per_T"].append({str(T):re_[T] for T in T_list})
      print(f"  E{epoch:3d}/{C['epochs']} | L={avg_loss:.4f} | P={ap:.2f} | EMA={ea:.2f}/{eas:.4f} "
            f"| LR={lr_now:.2e} | {time.time()-t_start:.0f}s")
      print("    EMA per-T: "+" ".join(f"T{T}={re_[T][0]:.2f}" for T in T_list))
      print("    batches per-T: "+" ".join(f"T{T}={t_count[T]}" for T in T_list))
      print("    ["+" ".join(f"{k}={v/max(1,nbat):.4f}" for k,v in detail.items() if k!="total")+"]")
      if ea>best_ema_psnr:
        best_ema_psnr=ea;best_epoch=epoch
        torch.save({"epoch":epoch,"model_state_dict":ema_model.state_dict(),
                    "ema_psnr":ea,"ema_ssim":eas,"per_T":re_,
                    "mask":cup.mask.detach().cpu(),"config":cfg_safe()},
                   os.path.join(C["output_dir"],"best_model.pth"))
        print(f"    ** New best mean EMA PSNR={ea:.2f} (epoch {epoch})")

    if epoch%20==0 or epoch==C["epochs"]:
      torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                  "optimizer_state_dict":optimizer.state_dict(),
                  "ema_model_state_dict":ema_model.state_dict(),
                  "mask":cup.mask.detach().cpu(),"config":cfg_safe()},
                 os.path.join(C["output_dir"],"checkpoint.pth"))

  train_time=time.time()-t_start
  model.eval();ema_model.eval()
  mask_w_final=eval_mask()
  torch.save(model.state_dict(),os.path.join(C["output_dir"],"dd_madun_tfree_model.pth"))
  torch.save({"model_state_dict":ema_model.state_dict(),"mask":cup.mask.detach().cpu(),
              "config":cfg_safe()},os.path.join(C["output_dir"],"dd_madun_tfree_ema.pth"))
  print(f"\n[4/5] Saved. Best mean EMA PSNR {best_ema_psnr:.2f} at epoch {best_epoch}, {train_time/3600:.1f}h")

  print("[5/5] Final evaluation (EMA):")
  final=evaluate(ema_model,mask_w_final,verbose=True)
  with open(os.path.join(C["output_dir"],"history.json"),"w") as f:
    json.dump({"history":history,"config":C,"final_per_T":{str(k):v for k,v in final.items()},
               "best_ema_psnr":best_ema_psnr,"best_epoch":best_epoch,
               "train_time_s":train_time},f,indent=2)
  pr=model.get_p()
  print(f"  sigma: {pr['sg']}\n  rho_f: {pr['rf']}\n  rho_s: {pr['rs']}")
  print(f"  Output: {os.path.abspath(C['output_dir'])}")

if __name__=="__main__":
  main()
