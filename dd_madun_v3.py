"""DD-MADUN v3: P0 Improvements (CBAM + freq band + edge loss)"""
import torch,torch.nn as nn,torch.nn.functional as F,numpy as np
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
  def __init__(self,in_c,nf=96,nb=8):
    super().__init__()
    self.head=nn.Sequential(nn.Conv2d(in_c+1,nf,3,1,1,bias=False),nn.ReLU(True))
    self.gz=nn.Conv2d(nf*2,nf,3,1,1,bias=True)
    self.gr=nn.Conv2d(nf*2,nf,3,1,1,bias=True)
    self.gh=nn.Conv2d(nf*2,nf,3,1,1,bias=True)
    self.blocks=nn.ModuleList([CBAMResBlock(nf) for _ in range(nb)])
    self.sf=nn.Conv2d(nf*2,nf,1,bias=False)
    self.tail=nn.Conv2d(nf,in_c,3,1,1,bias=False)
  def forward(self,x,sig,hp=None):
    B,T,H,W=x.shape
    if sig.dim()<=2:sig=sig.view(-1,1,1,1).expand(B,1,H,W)
    sig=sig.to(x.device)
    hf=self.head(torch.cat([x,sig],dim=1))
    if hp is None:hp=torch.zeros(B,hf.shape[1],H,W,device=hf.device)
    gi=torch.cat([hf,hp],dim=1)
    zg=torch.sigmoid(self.gz(gi))
    rg=torch.sigmoid(self.gr(gi))
    hc=torch.tanh(self.gh(torch.cat([hf,rg*hp],dim=1)))
    h=(1-zg)*hp+zg*hc
    feat=h
    for b in self.blocks:feat=b(feat)
    fused=self.sf(torch.cat([hf,feat],dim=1))
    return x-self.tail(fused),h
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
  def __init__(self,T=8,H=None,W=None,K=8,K1=3,ng=6,nf=96,nb=8):
    super().__init__()
    self.T=T;self.H=H;self.W=W;self.K=K;self.K1=K1;self.K2=K-K1
    self.register_buffer('dm',build_dct_matrix(T))
    self.lrs=nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(K)])
    self.lrf=nn.ParameterList([nn.Parameter(torch.tensor(-0.5)) for _ in range(K)])
    self.llf=nn.ParameterList([nn.Parameter(torch.tensor(-0.5)) for _ in range(K)])
    self.lmu=nn.ParameterList([nn.Parameter(torch.tensor(2.0)) for _ in range(K)])
    self.let=nn.ParameterList([nn.Parameter(torch.tensor(-1.5)) for _ in range(K)])
    self.lsg=nn.ParameterList([nn.Parameter(torch.tensor(-1.0)) for _ in range(self.K2)])
    self.lbt=nn.ParameterList([nn.Parameter(torch.tensor(-1.0)) for _ in range(self.K2)])
    self.dc=nn.ModuleList([DataConsistencyMomentum(ng) for _ in range(K)])
    self.den=nn.ModuleList([FFDNetGRUv3(T,nf,nb) for _ in range(self.K2)])
    self._iw()
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
  def forward(self,E,fwd,adj,mask_w=None):
    B=E.shape[0]
    x=adj(E)
    if mask_w is not None:
      # Plan B init: dark regions (mask_w~0) start from the bright-region mean
      # instead of all-zero, giving the denoiser a sane starting point so the
      # near-dark dead-zone is filled by the image prior, not left black.
      xb=x*mask_w
      msum=mask_w.sum(dim=[2,3],keepdim=True).clamp(min=1e-6)
      mmean=xb.sum(dim=[2,3],keepdim=True)/msum   # (B,1,1,1)
      x=mask_w*x+(1.0-mask_w)*mmean
    zs=x.clone();us=torch.zeros_like(x)
    zf=dct_along_time(x,self.dm);uf=torch.zeros_like(zf)
    v=torch.zeros_like(x);h=None;inter=[]
    for k in range(self.K):
      rs=torch.exp(self.lrs[k]);rf=torch.exp(self.lrf[k])
      lf=torch.exp(self.llf[k]);mk=torch.sigmoid(self.lmu[k]);ek=torch.exp(self.let[k])
      x,v=self.dc[k](x,v,zs,us,zf,uf,E,fwd,adj,self.dm,rs,rf,mk,ek,mask_w)
      vs=x+us
      if k<self.K1:
        ts=1.0/(rs+1e-8)
        zs=torch.sign(vs)*torch.relu(torch.abs(vs)-ts)
      else:
        idx=k-self.K1;sk=torch.exp(self.lsg[idx]);bk=torch.exp(self.lbt[idx])
        nm=self._anm(E,sk,bk,W_rec=x.shape[-1]);zs,h=self.den[idx](vs,nm,h)
      vf=dct_along_time(x,self.dm)+uf
      tf=rf/(lf+1e-8)
      zf=torch.sign(vf)*torch.relu(torch.abs(vf)-tf)
      us=us+x-zs;uf=uf+dct_along_time(x,self.dm)-zf
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
def freq_band_l1(p,g,dm,T=8):
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
      lf=freq_band_l1(Ip,Ig,self.dm); tl+=self.g*lf; ld["fb"]=lf.item()
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