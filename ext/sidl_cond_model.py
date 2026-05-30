"""
SIDLCondModel — ImageRestorationModel that forwards the degradation-type id to
a FiLM-conditioned network (config D / E).

Only feed_data is overridden: it stores the per-sample degradation id on the
(bare) network via set_deg(), so the inherited optimize_parameters / test paths
(which call self.net_g(self.lq)) work unchanged. This means config E simply
combines synth_aug + PSNRFFTLoss + this model + NAFNetFiLMLocal via YAML, with
no extra code.

Drop this file into  NAFNet/basicsr/models/  (auto-imported as *_model.py).
Use in YAML:  model_type: SIDLCondModel
"""

from basicsr.models.image_restoration_model import ImageRestorationModel


class SIDLCondModel(ImageRestorationModel):
    def feed_data(self, data, is_val=False):
        super().feed_data(data, is_val=is_val)
        deg = data.get('degradation_type', None)
        if deg is not None:
            net = self.net_g
            net = net.module if hasattr(net, 'module') else net
            if hasattr(net, 'set_deg'):
                net.set_deg(deg.to(self.device))
