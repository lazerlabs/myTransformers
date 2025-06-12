import os
import sys
import torch

# Add iTransformer model path to sys.path for importing models
iTransformer_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'iTransformer', 'model'))
if iTransformer_model_path not in sys.path:
    sys.path.insert(0, iTransformer_model_path)

# Import models from iTransformer directory
try:
    from Transformer import Model as Transformer
    from Informer import Model as Informer
    from Reformer import Model as Reformer
    from Flowformer import Model as Flowformer
    from Flashformer import Model as Flashformer
    from iTransformer import Model as iTransformer
    from iInformer import Model as iInformer
    from iReformer import Model as iReformer
    from iFlowformer import Model as iFlowformer
    from iFlashformer import Model as iFlashformer
except ImportError as e:
    print(f"Warning: Could not import some models from iTransformer directory: {e}")
    # Set to None for models that couldn't be imported
    Transformer = Informer = Reformer = Flowformer = Flashformer = None
    iTransformer = iInformer = iReformer = iFlowformer = iFlashformer = None


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # Check if CUDA is available
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            # Check if Apple Silicon MPS is available
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                print('Use Apple Silicon GPU: mps')
            else:
                device = torch.device('cpu')
                print('GPU requested but not available. Using CPU')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass 
