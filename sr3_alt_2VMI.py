import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import functools
import string

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """1x1 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3

class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')
    
def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 3, 1, 2)

class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij',  q,k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

class Conv2d(nn.Module):
  """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

  def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
               resample_kernel=(1, 3, 3, 1),
               use_bias=True,
               kernel_init=None):
    super().__init__()
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
    if kernel_init is not None:
      self.weight.data = kernel_init(self.weight.data.shape)
    if use_bias:
      self.bias = nn.Parameter(torch.zeros(out_ch))

    self.up = up
    self.down = down
    self.resample_kernel = resample_kernel
    self.kernel = kernel
    self.use_bias = use_bias

  def forward(self, x):
    if self.up:
      x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
    elif self.down:
      x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
    else:
      x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

    if self.use_bias:
      x = x + self.bias.reshape(1, -1, 1, 1)

    return x

class Upsamplepp(nn.Module):
  """ fir is currently being ignored. Hence, we skip adjustment 1 in Song et al. 2021"""
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        self.Conv2d_0 = Conv2d(in_ch, out_ch,
                             kernel=3, up=True,
                             resample_kernel=fir_kernel,
                             use_bias=True,
                             kernel_init=default_init())
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      if not self.with_conv:
        h = upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = self.Conv2d_0(x)

    return h

class Downsamplepp(nn.Module):
  """ fir is currently being ignored. Hence, we skip adjustment 1 in Song et al. 2021"""
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, down=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      if self.with_conv:
        x = F.pad(x, (0, 1, 0, 1))
        x = self.Conv_0(x)
      else:
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      if not self.with_conv:
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = self.Conv2d_0(x)

    return x

class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0.):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))
    h = self.Conv_0(h)
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.) 

# because with skip fir
def naive_upsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H, 1, W, 1))
  x = x.repeat(1, 1, 1, factor, 1, factor)
  return torch.reshape(x, (-1, C, H * factor, W * factor))

def naive_downsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
  return torch.mean(x, dim=(3, 5))

class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))
    
    if self.up:
      if self.fir: # never used due to issues with cpp
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = naive_upsample_2d(h, factor=2)
        x = naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = naive_downsample_2d(h, factor=2)
        x = naive_downsample_2d(x, factor=2)

    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

class NCSNpp(nn.Module):
    """ "Simplified" version of NCSN++/DDPM++ model (with additional global skip connection) """
    def __init__(self,
                num_channels = 2,
                output_channels = None,
                nf = 16,
                ch_mult = (1, 2, 4, 8, 16, 32, 32, 32),
                attn_resolutions = (16,),
                num_resblocks = 2,
                resamp_with_conv = True,
                skip_rescale = True,
                resblock_type = 'biggan',
                attention_type = 'ddpm',
                dropout = 0,
                init_scale = 0,
                image_size = 256,
                skip = False
                    ):
        super().__init__()
        self.act = act = nn.SiLU()
        self.nf = nf
        self.num_resblocks = num_resblocks 
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type 
        self.skip = skip

        modules = []
        AttnBlock = functools.partial(AttnBlockpp,
                                          init_scale=init_scale,
                                          skip_rescale=skip_rescale)
        Upsample = functools.partial(Upsamplepp,
                                     with_conv=resamp_with_conv)

        Downsample = functools.partial(Downsamplepp,
                                       with_conv=resamp_with_conv)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPMpp,
                                              act=act,
                                              dropout=dropout,
                                              init_scale=init_scale,
                                              skip_rescale=skip_rescale,
                                              temb_dim=None)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGANpp,
                                              act=act,
                                              dropout=dropout,
                                              init_scale=init_scale,
                                              skip_rescale=skip_rescale,
                                              temb_dim=None)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block
        channels = num_channels
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_resblocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))    

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_resblocks+1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch+hs_c.pop(),
                                          out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions: 
                modules.append(AttnBlock(channels=in_ch))

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(up=True, in_ch=in_ch))

        assert not hs_c
        
        
        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
        modules.append(conv3x3(in_ch, output_channels, init_scale=init_scale))
        self.all_modules = nn.ModuleList(modules)
        
    def forward(self, x):
        n_skip = x.clone()
        modules = self.all_modules
        m_idx = 0

        # Downsampling block 
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_resblocks):
                h = modules[m_idx](hs[-1])
                m_idx += 1

                if self.all_resolutions[i_level] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx +=1
                    
                hs.append(h)
                
            if i_level != self.num_resolutions -1:
                h = modules[m_idx](hs[-1])
                m_idx += 1
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1 
        
        # Upsampling block 
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_resblocks+1):
                h = modules[m_idx](torch.cat([h, hs.pop()],dim=1))
                m_idx += 1
                
            if self.all_resolutions[i_level] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1
                
        assert not hs
        
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)
                                 
        if self.skip:
            return n_skip + h
        else:
            return h

class NCSNpp_critic(nn.Module):
    """ "Simplified" version of NCSN++/DDPM++ model (with additional global skip connection) """
    def __init__(self,
                num_channels = 2,
                nf = 16,
                ch_mult = (1, 2, 4, 8, 16, 32, 32, 32),
                attn_resolutions = (16,),
                num_resblocks = 2,
                resamp_with_conv = True,
                skip_rescale = True,
                resblock_type = 'biggan',
                attention_type = 'ddpm',
                dropout = 0,
                init_scale = 0,
                image_size = 256
                    ):
        super().__init__()
        self.act = act = nn.SiLU()
        self.nf = nf
        self.num_resblocks = num_resblocks 
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type 

        modules = []
        AttnBlock = functools.partial(AttnBlockpp,
                                          init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Downsample = functools.partial(Downsamplepp,
                                       with_conv=resamp_with_conv)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPMpp,
                                              act=act,
                                              dropout=dropout,
                                              init_scale=init_scale,
                                              skip_rescale=skip_rescale,
                                              temb_dim=None)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGANpp,
                                              act=act,
                                              dropout=dropout,
                                              init_scale=init_scale,
                                              skip_rescale=skip_rescale,
                                              temb_dim=None)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block
        channels = num_channels
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_resblocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))    
        
        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(1,1))
        self.all_modules = nn.ModuleList(modules)
        
    def forward(self, x):
        modules = self.all_modules
        m_idx = 0

        # Downsampling block 
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_resblocks):
                h = modules[m_idx](hs[-1])
                m_idx += 1

                if self.all_resolutions[i_level] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx +=1
                    
                hs.append(h)
                
            if i_level != self.num_resolutions -1:
                h = modules[m_idx](hs[-1])
                m_idx += 1
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1 
                         
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)
                         
        return h
