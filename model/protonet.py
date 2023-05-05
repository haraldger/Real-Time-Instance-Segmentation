import torch.nn.functional as F
import numpy as np
import torch
from torch import nn

class InterpolateModule(nn.Module):
	"""
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

	def __init__(self, *args, **kwdargs):
		super().__init__()

		self.args = args
		self.kwdargs = kwdargs

	def forward(self, x):
		return F.interpolate(x, *self.args, **self.kwdargs)

class InterpolateModule(nn.Module):
	"""
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

	def __init__(self, *args, **kwdargs):
		super().__init__()

		self.args = args
		self.kwdargs = kwdargs

	def forward(self, x):
		return F.interpolate(x, *self.args, **self.kwdargs)
	

def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """
    def make_layer(layer_cfg):
        nonlocal in_channels
        
        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                #layer = Concat([net[0] for net in nets], layer_cfg[2])
                #num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        
        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels


#testing
def proto_test_shape():
  conf=[(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]
  proto,mask=make_net(256, conf, include_last_relu=True)
  x = torch.randn(2, 256, 69, 69) #shape of C3
  y=proto(x)
  print("shape of the protonet output is: ",y.shape)
  assert (y.shape==torch.Size([2, 32, 138, 138]))

def run_tests():
     proto_test_shape()
