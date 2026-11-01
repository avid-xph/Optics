"""sci_ablation: SCI/CUP 重建质量影响因素消融实验框架.

实验矩阵（PPT 第 4 页任务）:
  exp=T     序列深度 T ∈ {8,12,16,20,24,28,32}
  exp=cell  编码单元大小 c ∈ {1..10}（c×c 像元共享一个随机编码值）
  exp=joint 编码单元 × T 联合作用
  exp=noise 测量噪声 std ∈ {0, 0.005, 0.01, 0.02, 0.05, 0.1}

三类方法对比:
  iterative  TwIST / ADMM-L1 / ADMM-TV（对 H, W, T 不敏感）
  ddmadun    DD-MADUN v3 deep unfolding（对 T 敏感, 现有权重仅 T=8）
  e2e        End-to-end 网络（对 H, W, T 敏感, 需按 T 训练）
"""
