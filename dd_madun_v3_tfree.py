"""DD-MADUN v3: P0 Improvements (CBAM + freq band + edge loss)"""
import torch,torch.nn as nn,torch.nn.functional as F,numpy as np
import torch.utils.checkpoint
T_REF=8   # T the legacy (warm-start) weights were trained on; see DD_MADUNv3.forward
def build_dct_matrix(T):
  n=np.arange(T);k=np.arange(T).reshape(-1,1)
  D=np.cos(np.pi/T*k*(n+0.5));D[0]/=np.sqrt(2)
  return torch.from_numpy((np.sqrt(2.0/T)*D).astype(np.float32))
def dct_along_time(x,dm):
  B,T,H,W=x.shape;xt=x.permute(0,2,3,1).reshape(-1,T)
  return (xt@dm.T.to(x.device)).reshape(B,H,W,T).permute(0,3,1,2)
def idct_along_time(xd,dm):
  B,T,H,W=xd.shape;xt=xd.permute(0,2,3,1).reshape(-1,T)
  return (xt@dm.to(xd.device)).reshape(B,H,W,T).permute(0,3,1,2)
class CBAMBlock(nn.Module):
  def __init__(self,ch,r=8):
    super().__init__()
    r2=max(ch//r,4)
    self.fc=nn.Sequential(nn.Conv2d(ch,r2,1,bias=False),nn.ReLU(True),nn.Conv2d(r2,ch,1,bias=False))
    self.sc=nn.Conv2d(2,1,7,padding=3,bias=False)
  def forward(self,x):
    a=self.fc(F.adaptive_avg_pool2d(x,1))
    m=self.fc(F.adaptive_max_pool2d(x,1))
    x=x*torch.sigmoid(a+m)
    a2=x.mean(1,keepdim=True);m2,_=x.max(1,keepdim=True)
    return x*torch.sigmoid(self.sc(torch.cat([a2,m2],dim=1)))
class CBAMResBlock(nn.Module):
  def __init__(self,nf,r=8):
    super().__init__()
    self.c1=nn.Conv2d(nf,nf,3,padding=1,bias=False)
    self.c2=nn.Conv2d(nf,nf,3,padding=1,bias=False)
    self.relu=nn.ReLU(True);self.cbam=CBAMBlock(nf,r)
  def forward(self,x):
    r=x;x=self.relu(self.c1(x));x=self.c2(x)
    x=self.cbam(x);return self.relu(x+r)
class FFDNetGRUv3(nn.Module):
  """T-free denoiser: frame-wise conv, weights never depend on T.

  temporal:
    'none' - A2: each frame processed independently (parallel, cheapest)
    'gru'  - A1: GRU unrolled along the time axis (cross-frame modelling)
  Both variants share identical parameter shapes, so weights are interchangeable.
  """
  def __init__(self,nf=96,nb=8,temporal='none',ckpt=False):
    super().__init__()
    self.temporal=temporal
    self.use_ckpt=bool(ckpt)   # per-CBAMResBlock checkpointing, see forward()
    self.head=nn.Sequential(nn.Conv2d(2,nf,3,1,1,bias=False),nn.ReLU(True))
    self.gz=nn.Conv2d(nf*2,nf,3,1,1,bias=True)
    self.gr=nn.Conv2d(nf*2,nf,3,1,1,bias=True)
    self.gh=nn.Conv2d(nf*2,nf,3,1,1,bias=True)
    self.blocks=nn.ModuleList([CBAMResBlock(nf) for _ in range(nb)])
    self.sf=nn.Conv2d(nf*2,nf,1,bias=False)
    self.tail=nn.Conv2d(nf,1,3,1,1,bias=False)
  def _gru(self,hf,hp):
    gi=torch.cat([hf,hp],dim=1)
    zg=torch.sigmoid(self.gz(gi))
    rg=torch.sigmoid(self.gr(gi))
    hc=torch.tanh(self.gh(torch.cat([hf,rg*hp],dim=1)))
    return (1-zg)*hp+zg*hc
  def forward(self,x,sig,hp=None):
    B,T,H,W=x.shape
    if sig.dim()<=2:sig=sig.view(-1,1,1,1).expand(B,1,H,W)
    sig=sig.to(x.device)
    # frame-wise head: (B*T,2,H,W) -> (B*T,nf,H,W)
    xs=x.reshape(B*T,1,H,W)
    sgs=sig.unsqueeze(1).expand(B,T,1,H,W).reshape(B*T,1,H,W)
    hf=self.head(torch.cat([xs,sgs],dim=1))
    nf=hf.shape[1]
    hf5=hf.view(B,T,nf,H,W)
    if hp is None:hp=torch.zeros(B,nf,H,W,device=hf.device)
    if self.temporal=='gru':
      h=hp;hs=[]
      for t in range(T):
        h=self._gru(hf5[:,t],h);hs.append(h)
      feat=torch.stack(hs,dim=1).reshape(B*T,nf,H,W);h_out=h
    else:
      hpb=hp.unsqueeze(1).expand(B,T,nf,H,W).reshape(B*T,nf,H,W)
      feat=self._gru(hf,hpb)
      h_out=feat.view(B,T,nf,H,W)[:,-1]
    do_ckpt=self.use_ckpt and torch.is_grad_enabled()
    for b in self.blocks:
      feat=torch.utils.checkpoint.checkpoint(b,feat,use_reentrant=False) if do_ckpt else b(feat)
    fused=self.sf(torch.cat([hf,feat],dim=1))
    return x-self.tail(fused).reshape(B,T,H,W),h_out
class DataConsistencyMomentum(nn.Module):
  def __init__(self,ng=6):
    super().__init__();self.ng=ng
  def forward(self,x,v,zs,us,zf,uf,E,fwd,adj,dm,rs,rf,mk,ek,mask_w=None):
    # mask_w: (1,1,H,W) in [0,1], ~1 in bright regions, ~0 in near-dark regions.
    # When given, the data-fidelity gradient is gated so dark regions are NOT
    # pulled by the measurement (A^T A ~ M^2 ~ 0 there anyway) and are left
    # entirely to the denoiser / image prior (Plan B: mask-weighted DC).
    for _ in range(self.ng):
      xh=x+mk*v
      gd=adj(fwd(xh)-E)
      if mask_w is not None:
        gd=gd*mask_w
      gp=rs*(xh-zs+us)
      fx=dct_along_time(xh,dm)
      gf=idct_along_time(rf*(fx-zf+uf),dm)
      g=gd+gp+gf;v=mk*v+ek*g;x=xh-v
    return x,v
class DD_MADUNv3(nn.Module):
  def __init__(self,T=8,H=None,W=None,K=8,K1=3,ng=6,nf=96,nb=8,temporal='none',
              use_checkpoint=False,ckpt_blocks=False,final_prox=True):
    super().__init__()
    self.T=T;self.H=H;self.W=W;self.K=K;self.K1=K1;self.K2=K-K1
    self.temporal=temporal
    # Two independently controllable checkpoint granularities (see forward()/
    # _stage() and FFDNetGRUv3.forward()). Plain bools, not Parameters/
    # buffers, so neither is written into state_dict and both can be freely
    # toggled after construction or after loading a checkpoint.
    #   use_checkpoint : per unfolding stage (cheap: K=8 recompute segments,
    #                    each stage's own compute is modest outside its
    #                    denoiser call). Start with this alone.
    #   ckpt_blocks    : ALSO per CBAMResBlock inside each denoiser (nb
    #                    extra recompute segments per denoiser stage, nested
    #                    inside the stage-level one). Much more aggressive
    #                    memory savings but noticeably slower -- only turn
    #                    this on if use_checkpoint alone still OOMs.
    self.use_ckpt=bool(use_checkpoint)
    # Return the final PROX output (zs) instead of the final DC output (x).
    # Setting this False restores the legacy behavior, which is ONLY useful for
    # evaluating checkpoints trained with the old code: in those, den[K2-1] was
    # never in the loss graph (zero gradient) and is still at random init, so
    # routing the output through it makes them worse. New training must use True.
    self.final_prox=bool(final_prox)
    self._dm_cache={}   # T -> dct matrix; plain dict so it stays out of state_dict
    self.lrs=nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(K)])
    self.lrf=nn.ParameterList([nn.Parameter(torch.tensor(-0.5)) for _ in range(K)])
    self.llf=nn.ParameterList([nn.Parameter(torch.tensor(-0.5)) for _ in range(K)])
    self.lmu=nn.ParameterList([nn.Parameter(torch.tensor(2.0)) for _ in range(K)])
    self.let=nn.ParameterList([nn.Parameter(torch.tensor(-1.5)) for _ in range(K)])
    self.lsg=nn.ParameterList([nn.Parameter(torch.tensor(-1.0)) for _ in range(self.K2)])
    self.lbt=nn.ParameterList([nn.Parameter(torch.tensor(-1.0)) for _ in range(self.K2)])
    self.dc=nn.ModuleList([DataConsistencyMomentum(ng) for _ in range(K)])
    self.den=nn.ModuleList([FFDNetGRUv3(nf,nb,temporal=temporal,ckpt=ckpt_blocks) for _ in range(self.K2)])
    self._iw()
  def _get_dm(self,T,device):
    """DCT matrix for a given T, built on demand and cached (never stored in state_dict)."""
    key=(T,str(device))
    if key not in self._dm_cache:
      self._dm_cache[key]=build_dct_matrix(T).to(device)
    return self._dm_cache[key]
  def set_checkpointing(self,stage=None,blocks=None):
    """Toggle checkpointing granularities at runtime, e.g. after loading a
    checkpoint trained without it. stage=True/False controls per-unfolding-
    stage checkpointing; blocks=True/False controls the (more aggressive,
    slower) per-CBAMResBlock checkpointing nested inside each denoiser.
    Pass only the one(s) you want to change; None leaves it as-is. No effect
    on state_dict / architecture."""
    if stage is not None:
      self.use_ckpt=bool(stage)
    if blocks is not None:
      for d in self.den:
        d.use_ckpt=bool(blocks)
    return self
  def _iw(self):
    for m in self.modules():
      if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        if m.bias is not None:nn.init.zeros_(m.bias)
  def _anm(self,E,sk,bk,W_rec=None):
    """adaptive noise map from measurement E; crop to reconstruction width W_rec."""
    B,H,Wm=E.shape
    gx=torch.abs(E-torch.roll(E,1,-1));gx[:,:,0]=gx[:,:,1]
    gm=gx.amin((1,2),True);gx2=gx.amax((1,2),True)
    gn=torch.where(gx2>gm+1e-8,(gx-gm)/(gx2-gm),torch.zeros_like(gx))
    nm=(sk*(1.0+bk*gn)).unsqueeze(1)        # (B,1,H,Wm)
    Wc = W_rec if W_rec is not None else self.W
    return nm[:,:,:,:Wc]                      # (B,1,H,W_rec)
  def _stage(self,k,x,v,zs,us,zf,uf,E,fwd,adj,dm,mask_w,h):
    """One unfolding stage: DC block (ng grad steps) + sparsity-or-denoiser
    prox + frequency-domain prox. Factored out of forward() so it can be
    wrapped in gradient checkpointing PER STAGE (see forward()) -- this is
    what makes ckpt=True actually reduce peak memory for large T. Wrapping
    the whole K-stage loop in a single outer checkpoint call would still
    force every stage's (including all K2 independent denoisers') activations
    to be materialized simultaneously during the one recompute+backward pass,
    giving no real memory benefit over not checkpointing at all."""
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
    vf=dct_along_time(x,dm)+uf
    tf=rf/(lf+1e-8)
    zf=torch.sign(vf)*torch.relu(torch.abs(vf)-tf)
    us=us+x-zs;uf=uf+dct_along_time(x,dm)-zf
    return x,v,zs,us,zf,uf,h
  def forward(self,E,fwd,adj,mask_w=None):
    B=E.shape[0]
    x=adj(E)
    T=x.shape[1]                       # T is dictated by the operator, not by self.T
    # ||A^T A|| grows ~linearly with T (measured 8.0/15.5/20.0/25.1 at T=8/16/24/32),
    # so DC step sizes learned at T_REF diverge at larger T. Scale the adjoint by
    # T_REF/T: the data-term Lipschitz constant stays at the T_REF level and
    # T=T_REF is bit-identical to the legacy model.
    s=T_REF/T
    if s!=1.0:
      adj=lambda r,_adj=adj:_adj(r)*s
      x=x*s
    dm=self._get_dm(T,x.device)        # DCT matrix matched to this T
    if mask_w is not None:
      # Plan B init: dark regions (mask_w~0) start from the bright-region mean
      # instead of all-zero, giving the denoiser a sane starting point so the
      # near-dark dead-zone is filled by the image prior, not left black.
      xb=x*mask_w
      msum=mask_w.sum(dim=[2,3],keepdim=True).clamp(min=1e-6)
      mmean=xb.sum(dim=[2,3],keepdim=True)/msum   # (B,1,1,1)
      x=mask_w*x+(1.0-mask_w)*mmean
    zs=x.clone();us=torch.zeros_like(x)
    zf=dct_along_time(x,dm);uf=torch.zeros_like(zf)
    v=torch.zeros_like(x);inter=[]
    if self.K2>0:
      # Initialize the cross-stage denoiser state as a real zero tensor
      # (rather than None) so every stage's _stage() call has a uniform,
      # checkpoint-friendly tensor signature; numerically identical to the
      # old None->zeros fallback inside FFDNetGRUv3.forward.
      nf0=self.den[0].head[0].out_channels
      h=torch.zeros(B,nf0,x.shape[-2],x.shape[-1],device=x.device,dtype=x.dtype)
    else:
      h=None
    do_ckpt=self.use_ckpt and torch.is_grad_enabled()
    for k in range(self.K):
      if do_ckpt:
        x,v,zs,us,zf,uf,h=torch.utils.checkpoint.checkpoint(
          self._stage,k,x,v,zs,us,zf,uf,E,fwd,adj,dm,mask_w,h,
          use_reentrant=False)
      else:
        x,v,zs,us,zf,uf,h=self._stage(k,x,v,zs,us,zf,uf,E,fwd,adj,dm,mask_w,h)
      inter.append(x.clone())
    # Return the LAST PROX output (zs), not the last DC output (x).
    #
    # The prox of the final stage only feeds us/uf, which no later stage ever
    # consumes -- so if we return x, den[K2-1] (and lsg/lbt of that stage) sits
    # completely outside the loss graph and receives exactly ZERO gradient
    # (measured: 20% of all parameters dead at K=8,K1=3, i.e. one of the five
    # denoisers stays at random init and is unused at inference too).
    # Returning zs -- the ADMM primal solution, denoiser output for k >= K1 --
    # puts every stage back into the graph. Verified: dead params 20% -> 0.
    if self.final_prox:
      return zs,inter
    return x,inter
  def get_p(self):
    return {'rs':[torch.exp(p).item() for p in self.lrs],
            'rf':[torch.exp(p).item() for p in self.lrf],
            'lf':[torch.exp(p).item() for p in self.llf],
            'mu':[torch.sigmoid(p).item() for p in self.lmu],
            'et':[torch.exp(p).item() for p in self.let],
            'sg':[torch.exp(p).item() for p in self.lsg],
            'bt':[torch.exp(p).item() for p in self.lbt]}
def freq_band_l1(p,g,dm,T=None):
  if T is None:T=dm.shape[0]      # infer T from the DCT matrix (was hard-coded to 8)
  pd=dct_along_time(p,dm);gd=dct_along_time(g,dm)
  nl=max(1,T//4);nm=max(2,T//2);nh=T-nl-nm
  d=(pd-gd).abs()
  ll=d[:,:nl].mean();lm=d[:,nl:nl+nm].mean();lh=d[:,nl+nm:].mean() if nh>0 else 0.0
  return 1.0*ll+2.0*lm+0.5*lh
def edge_loss(p,g):
  dxp=p[:,:,:,1:]-p[:,:,:,:-1];dxg=g[:,:,:,1:]-g[:,:,:,:-1]
  dyp=p[:,:,1:,:]-p[:,:,:-1,:];dyg=g[:,:,1:,:]-g[:,:,:-1,:]
  return F.l1_loss(dxp,dxg)+F.l1_loss(dyp,dyg)
def ssim_loss(p,g,ws=7,c1=0.0001,c2=0.0009):
  pad=ws//2;B,T,H,W=p.shape
  pf=p.reshape(B*T,1,H,W);gf=g.reshape(B*T,1,H,W)
  mp=F.avg_pool2d(pf,ws,1,pad);mg=F.avg_pool2d(gf,ws,1,pad)
  sp=F.avg_pool2d(pf*pf,ws,1,pad)-mp*mp
  sg=F.avg_pool2d(gf*gf,ws,1,pad)-mg*mg
  spg=F.avg_pool2d(pf*gf,ws,1,pad)-mp*mg
  sm=((2*mp*mg+c1)*(2*spg+c2))/((mp*mp+mg*mg+c1)*(sp+sg+c2))
  return 1.0-sm.mean()
"""B2: Focal Frequency Loss (Cai et al., ICCV 2021).
Spectral distance |F(pred) - F(gt)| re-weighted per-frequency by the current
error magnitude (detached), so hard high-frequency components dominate."""
class FocalFrequencyLoss(nn.Module):
  def __init__(self, alpha=1.0, eps=1e-8):
    super().__init__();self.alpha=alpha;self.eps=eps
  def forward(self, pred, target):
    # torch.fft.fft2 does NOT support bfloat16 (verified on RTX 5090 / torch
    # 2.12: "Unsupported dtype BFloat16"), so under bf16 autocast this used to
    # be a landmine -- it only survived because the fp32 CUP mask promotes the
    # model output back to fp32. Cast explicitly so AMP can never break it.
    pred = pred.float(); target = target.float()
    diff = torch.fft.fft2(pred, norm="ortho") - torch.fft.fft2(target, norm="ortho")
    d = torch.abs(diff)
    w = d.detach() ** self.alpha
    w = w / w.amax(dim=(-2,-1), keepdim=True).clamp(min=self.eps)
    return (w * d).mean()

"""B7: VGG perceptual loss using pretrained VGG16 features.
Computes L1 distance in VGG16 feature space (relu1_2, relu2_2, relu3_3)."""
class VGGPerceptualLoss(nn.Module):
  def __init__(self, device="cuda"):
    super().__init__()
    try:
      import torchvision.models as tvm
      vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_FEATURES).features
      # Slice to get relu1_2 (3), relu2_2 (8), relu3_3 (15)
      self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)]).eval()
      self.slice2 = nn.Sequential(*[vgg[i] for i in range(4,9)]).eval()
      self.slice3 = nn.Sequential(*[vgg[i] for i in range(9,16)]).eval()
      for p in self.parameters(): p.requires_grad_(False)
      self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
      self.register_buffer("std", torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
      self.to(device)
      self.available = True
    except Exception as e:
      print(f"  [VGGPerceptualLoss] VGG16 unavailable ({e}), using zero loss")
      self.available = False
  def _norm(self, x):
    # x: (B,T,H,W) grayscale -> repeat to 3-channel, normalize
    B,T,H,W = x.shape
    x3 = x.reshape(B*T,1,H,W).expand(B*T,3,H,W)
    return (x3 - self.mean.to(x.device)) / self.std.to(x.device)
  def forward(self, pred, target):
    if not self.available: return pred.new_zeros(())
    pn = self._norm(pred); tn = self._norm(target)
    f1p=self.slice1(pn); f2p=self.slice2(f1p); f3p=self.slice3(f2p)
    f1t=self.slice1(tn); f2t=self.slice2(f1t); f3t=self.slice3(f2t)
    return (F.l1_loss(f1p,f1t) + F.l1_loss(f2p,f2t) + F.l1_loss(f3p,f3t)) / 3.0

class DDMADUNLossv3(nn.Module):
  def __init__(self, as_=0.15, gf=0.15, sw=0.20, ew=0.05, dm=None,
               ffl_w=0.0, vgg_w=0.0, device="cuda"):
    super().__init__()
    self.a=as_; self.g=gf; self.sw=sw; self.ew=ew; self.dm=dm
    self.ffl_w=ffl_w; self.vgg_w=vgg_w
    self.ffl_loss = FocalFrequencyLoss(alpha=1.0) if ffl_w > 0 else None
    self.vgg_loss = VGGPerceptualLoss(device=device) if vgg_w > 0 else None
  def forward(self, Ip, Ig, inter=None):
    l1=F.l1_loss(Ip,Ig); tl=l1; ld={"l1":l1.item()}
    if inter is not None:
      K=len(inter); ls=0.0
      for k,xk in enumerate(inter): ls += ((k+1)/K)*F.l1_loss(xk,Ig)
      ls/=K; tl+=self.a*ls; ld["st"]=ls.item()
    if self.dm is not None and self.g>0:
      dm=self.dm
      if dm.shape[0]!=Ip.shape[1]:           # rebuild if T differs from the cached one
        dm=build_dct_matrix(Ip.shape[1]).to(Ip.device)
      lf=freq_band_l1(Ip,Ig,dm); tl+=self.g*lf; ld["fb"]=lf.item()
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