import warnings
from collections import OrderedDict, Iterable, Mapping
from itertools import islice
import operator
import torch
from ex_methods.module.module import Module
import torch.nn.functional as F


class Container(Module):

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        # DeprecationWarning is ignored by default <sigh>
        warnings.warn("nn.Container is deprecated. All of it's functionality "
                      "is now implemented in nn.Module. Subclass that instead.")
        for key, value in kwargs.items():
            self.add_module(key, value)


class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def reset_parameters(self):
        pass

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        # classic forward
        activation_output = input
        for module in self._modules.values():
            activation_output = module.forward(activation_output)
        return activation_output

    def accuracy(self, activation_output, labels):
        prediction = torch.argmax(activation_output, 1)
        c = (prediction == labels).float().squeeze()
        accuracy = torch.mean(c)
        return accuracy

    def class_loss(self, activation_output, labels):
        criterion = torch.nn.CrossEntropyLoss()
        diff = criterion(activation_output, labels)
        loss = torch.mean(diff)
        return loss

    def normalize(self, LRP):
        LRP = LRP - LRP.min(dim=1)[0].min(dim=1)[0].min(dim=1)[0].reshape(-1, 1, 1, 1)
        LRP_shape = LRP.shape
        LRP = LRP / (LRP.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].reshape(-1, 1, 1, 1) + 1e-8)
        return LRP

    def location(self, LRP, keep_batch=False, interpreter='lrp', lrp_target_layer=None):
        l = LRP.shape[2]
        n = LRP.shape[0]
        mask = torch.ones_like(LRP)
        k = (l // 7)
        if interpreter == 'lrp' and lrp_target_layer == None:
            LRP = LRP / LRP.max()
            mask[:, :, k:-k, k:-k] = -1 * torch.ones((l - 2 * k, l - 2 * k), dtype=torch.float32).cuda()
        else:
            mask[:, :, k:-k, k:-k] = torch.zeros((l - 2 * k, l - 2 * k), dtype=torch.float32).cuda()

        LRP = LRP.reshape(n, -1)
        mask = mask.reshape(n, -1)
        loss = ((mask - LRP) ** 2).mean(dim=1)

        if keep_batch:
            return loss
        loss = loss.mean()
        return loss

    def corner(self, LRP):
        LRP = self.normalize(LRP)
        l = LRP.shape[2]
        mask = torch.zeros_like(LRP)
        k = (l // 7) * 2
        mask[:, :, :k, :k] = torch.ones((k, k), dtype=torch.float32).cuda()
        loss = torch.nn.functional.mse_loss(LRP, mask)
        return loss

    def topk(self, LRP, LRP_label, keep_batch=False):
        n = LRP.shape[0]
        LRP = self.normalize(LRP).reshape(n, -1)
        LRP_label = self.normalize(LRP_label).reshape(n, -1)
        l = LRP.shape[1]

        _, topk_index = LRP_label.topk(l // 10)

        for i, ind in enumerate(topk_index):
            if i == 0:
                loss = ((LRP[i, ind]) ** 2).mean(dim=0, keepdim=True)
            else:
                loss_new = ((LRP[i, ind]) ** 2).mean(dim=0, keepdim=True)
                loss = torch.cat((loss, loss_new), 0)

        if keep_batch:
            return loss
        loss = loss.mean()
        return loss

    def center_mass(self, LRP, LRP_label, keep_batch=False):
        n, c, h, w = LRP.shape
        LRP = self.normalize(LRP)
        LRP_label = self.normalize(LRP_label).detach().clone()

        LRP_sum = LRP.sum(dim=(1, 2, 3))
        LRP_label_sum = LRP_label.sum(dim=(1, 2, 3))

        h_vector = torch.range(1, h).reshape(1, 1, h, 1).cuda()
        w_vector = torch.range(1, h).reshape(1, 1, 1, w).cuda()

        LRP_h = (LRP.sum(dim=3, keepdim=True) * h_vector).sum(dim=2).squeeze() / (LRP_sum + 1e-8)
        LRP_w = (LRP.sum(dim=2, keepdim=True) * w_vector).sum(dim=3).squeeze() / (LRP_sum + 1e-8)

        LRP_label_h = (LRP_label.sum(dim=3, keepdim=True) * h_vector).sum(dim=2).squeeze() / (LRP_label_sum + 1e-8)
        LRP_label_w = (LRP_label.sum(dim=2, keepdim=True) * w_vector).sum(dim=3).squeeze() / (LRP_label_sum + 1e-8)

        if keep_batch:
            distance = ((LRP_h - LRP_label_h) ** 2 + (LRP_w - LRP_label_w) ** 2)
            return ((distance / 2) ** 0.5) / (w - 1)

        distance = ((LRP_h - LRP_label_h).abs() + (LRP_w - LRP_label_w).abs()).mean()
        return -distance

    def forward_targeted(self, inputs, labels, net_ori, c1, c2, target_layer=None, interpreter='lrp',
                         r_method="composite"):
        labels_c1 = labels + c1
        labels_c2 = labels + c2

        if interpreter == 'grad_cam':
            activation_output = net_ori.prediction(inputs)
            LRP_ori_c1 = net_ori.grad_cam(activation_output, labels_c1, target_layer)
            LRP_ori_c2 = net_ori.grad_cam(activation_output, labels_c2, target_layer)
            LRP_ori_c1 = LRP_ori_c1.clone().detach()
            LRP_ori_c2 = LRP_ori_c2.clone().detach()

            activation_output = self.prediction(inputs)
            LRP_c1 = self.grad_cam(activation_output, labels_c1, target_layer)
            LRP_c2 = self.grad_cam(activation_output, labels_c2, target_layer)

        if interpreter == 'lrp_T':
            activation_output = net_ori.prediction(inputs)
            LRP_ori_c1 = net_ori.lrp(activation_output, labels_c1, r_method, 1e-8, None, target_layer=target_layer)
            LRP_ori_c2 = net_ori.lrp(activation_output, labels_c2, r_method, 1e-8, None, target_layer=target_layer)
            LRP_ori_c1 = LRP_ori_c1.clone().detach()
            LRP_ori_c2 = LRP_ori_c2.clone().detach()

            activation_output = self.prediction(inputs)
            LRP_c1 = self.lrp(activation_output, labels_c1, r_method, 1e-8, None, target_layer=target_layer)
            LRP_c2 = self.lrp(activation_output, labels_c2, r_method, 1e-8, None, target_layer=target_layer)

        if interpreter == 'simple_grad_T':
            activation_output = net_ori.prediction(inputs)
            LRP_ori_c1 = net_ori.simple_grad(activation_output, labels_c1, target_layer)
            LRP_ori_c2 = net_ori.simple_grad(activation_output, labels_c2, target_layer)
            LRP_ori_c1 = LRP_ori_c1.clone().detach()
            LRP_ori_c2 = LRP_ori_c2.clone().detach()

            activation_output = self.prediction(inputs)
            LRP_c1 = self.simple_grad(activation_output, labels_c1, target_layer)
            LRP_c2 = self.simple_grad(activation_output, labels_c2, target_layer)

        if len(LRP_c1.shape) != 4:
            LRP_c1.unsqueeze(1)
            LRP_c2.unsqueeze(1)
            LRP_ori_c1.unsqueeze(1)
            LRP_ori_c2.unsqueeze(1)

        LRP_c1 = LRP_c1.sum(dim=1, keepdim=True)
        LRP_c2 = LRP_c2.sum(dim=1, keepdim=True)
        LRP_ori_c1 = LRP_ori_c1.sum(dim=1, keepdim=True)
        LRP_ori_c2 = LRP_ori_c2.sum(dim=1, keepdim=True)

        lrp_loss = 0.5 * (
                    torch.nn.functional.mse_loss(LRP_c1, LRP_ori_c2) + torch.nn.functional.mse_loss(LRP_c2, LRP_ori_c1))

        return lrp_loss

    def forward_classic(self, input, labels):
        activation_output = self.prediction(input)

        # Class loss and score
        _, prediction = torch.max(activation_output, 1)
        criterion = torch.nn.CrossEntropyLoss()
        diff = criterion(activation_output, labels)

        class_loss = torch.mean(diff)

        return class_loss

    #def forward(self, inputs, labels, lambda_value=2, target_layer=None, LRP_ori=None, interpreter='lrp',
     #           lrp_target_layer=None, loss_type="location"):
      #  activation_output = self.prediction(inputs)
       # return activation_output

        # # Class loss and score
        # _, prediction = torch.max(activation_output, 1)
        # criterion = torch.nn.CrossEntropyLoss()
        # diff = criterion(activation_output, labels)
        #
        # class_loss = torch.mean(diff)
        #
        # LRP = self.interpretation(activation_output, interpreter=interpreter, labels=labels, target_layer=lrp_target_layer, inputs=inputs)
        #
        # if len(LRP.shape) != 4:
        #     LRP = LRP.unsqueeze(1)
        # LRP = LRP.sum(dim=1, keepdim=True)
        #
        # # Calculate loss
        # if loss_type == 'location':
        #     lrp_loss = self.location(LRP)
        # elif loss_type == 'topk':
        #     lrp_loss = self.topk(LRP, LRP_ori)
        # elif loss_type == 'center_mass':
        #     lrp_loss = self.center_mass(LRP, LRP_ori)
        #
        # else:
        #     print('wrong loss type!!')
        #
        # total_loss = class_loss + (lambda_value *lrp_loss)
        # check_total_loss = total_loss
        #
        # return total_loss, class_loss, lrp_loss, activation_output, check_total_loss, LRP

    def set_lrp_parameters(self, lrp_var=None, param=None):
        for module in self._modules.values():
            module.set_lrp_parameters(lrp_var=lrp_var, param=param)

    def interpretation(self, activation_output, interpreter, labels, num_target, device, target_layer=None, inputs=None,
                       r_method="composite"):
        if inputs is None and interpreter in ['smooth_grad', 'integrated_grad', 'smooth_grad_T', 'integrated_grad_T']:
            print('you need to give inputs!!')

        if interpreter == 'lrp':
            LRP = self.lrp(activation_output, labels=labels, lrp_var=r_method, param=1e-8, whichScore=None,
                           target_layer=None)

        elif interpreter == 'lrp_T':
            LRP = self.lrp(activation_output, labels=labels, lrp_var=r_method, param=1e-8, whichScore=None,
                           target_layer=target_layer)

        elif interpreter == 'grad_cam':
            LRP = self.grad_cam(activation_output, device, labels, num_target, target_layer)

        elif interpreter == 'simple_grad':
            LRP = self.simple_grad(activation_output, device, labels, num_target, None)

        elif interpreter == 'simple_grad_T':
            LRP = self.simple_grad(activation_output, device, labels, num_target, target_layer)

        elif interpreter == 'smooth_grad':
            LRP = self.smooth_grad(self, inputs, labels, None)

        elif interpreter == 'smooth_grad_T':
            LRP = self.smooth_grad(self, inputs, labels, target_layer, device)

        elif interpreter == 'integrated_grad':
            LRP = self.integrated_grad(self, inputs, labels, None, device)

        elif interpreter == 'integrated_grad_T':
            LRP = self.integrated_grad(self, inputs, labels, target_layer, device)

        else:
            print('wrong interpreter!!')

        if len(LRP.shape) != 4:
            LRP = LRP.unsqueeze(1)

        LRP = LRP.sum(dim=1, keepdim=True)
        return LRP

    def integrated_grad(self, net, inputs, labels, target_layer, device, by_label=True, num_target=1000, smooth_num=16):
        iterations = smooth_num
        inputs_zero = torch.zeros_like(inputs).to(device)
        for i in range(iterations):
            alpha = float(i) / iterations
            inputs_interpolation = (1 - alpha) * inputs_zero + alpha * inputs
            activation_output = self.prediction(inputs_interpolation)
            if i == 0:
                R = self.simple_grad(activation_output, device, labels, target_layer,num_target, integrated_grad=True).detach()
            else:
                # If you want to train the model by using LR with IG, then you need to remove detach() function.
                R += self.simple_grad(activation_output, device, labels, target_layer, num_target, integrated_grad=True).detach()
            R = R / iterations

        return R

    def smooth_grad(self, inputs, labels, target_layer, device, by_label=True, num_target=1000, alpha=0.01, smooth_num=16,
                    smooth_std=0.1):
        iterations = smooth_num
        alpha = smooth_std
        for i in range(iterations):
            inputs_noise = inputs + alpha * torch.randn(inputs.shape).to(device)
            activation_output = self.prediction(inputs_noise)
            if i == 0:
                R = self.simple_grad(activation_output, device, labels, target_layer,num_target).detach()
            else:
                # If you want to train the model by using LR with smooth grad, then you need to remove detach() function.
                R += self.simple_grad(activation_output, device, labels, target_layer,num_target).detach()

        return R / iterations

    def simple_grad(self, output, device, labels, target_layer, num_target, by_label=True, integrated_grad=False):
        eye = torch.eye(num_target, dtype=torch.float32)
        eye = eye.to(device)
        if by_label:
            dx = eye[labels]
        else:
            dx = eye[torch.argmax(output, dim=1)]

        dx = torch.ones_like(output) * dx

        for key, module in reversed(list(self._modules.items())):

            requires_activation = (key == target_layer)
            # x: feature map, dx: dL/dx
            dx, x = module.grad_cam(dx, requires_activation)

            if requires_activation:
                break

        if integrated_grad:
            return dx
        R = torch.abs(dx)
        w = torch.ones_like(R) * 2

        if target_layer is not None:
            R = torch.sum(R * w, dim=1)
            # thresholding is useless
            R = torch.nn.functional.threshold(R, threshold=0, value=0)
            R_max, _ = R.max(1, keepdim=True)
            R_max, _ = R_max.max(2, keepdim=True)
            R_min, _ = R.min(1, keepdim=True)
            R_min, _ = R_min.min(2, keepdim=True)

            # R \in [0,1]
            R = (R - R_min) / (R_max - R_min + 1e-8)

            return R

        return R / torch.sum(R, dim=[1, 2, 3], keepdim=True)

    def grad_cam(self, output, device, labels, num_target, target_layer, by_label=True):
        eye = torch.eye(num_target, dtype=torch.float32)
        eye = eye.to(device)
        if by_label:
            dx = eye[labels]

        # target gradient represented by one-hot
        dx = torch.ones_like(output) * dx

        for key, module in reversed(list(self._modules.items())):
            requires_activation = (key == target_layer)

            # x: feature map, dx: gradient
            dx, x = module.grad_cam(dx, requires_activation)
            if requires_activation:
                break

        # global average pooling
        weights = dx.mean(3, keepdim=True)
        weights = weights.mean(2, keepdim=True)

        # 加权求和
        cam = torch.sum(x * weights, dim=1)
        # RELU
        cam = torch.nn.functional.threshold(cam, threshold=0, value=0)

        cam_max, _ = cam.max(1, keepdim=True)
        cam_max, _ = cam_max.max(2, keepdim=True)
        cam_min, _ = cam.min(1, keepdim=True)
        cam_min, _ = cam_min.min(2, keepdim=True)
        cam = cam / (cam_max + 1e-8)

        return cam

    def lrp(self, R, labels=None, lrp_var="composite", param=None, whichScore=None, target_layer=None):
        '''
        Performs LRP by calling subroutines, depending on lrp_var and param or
        preset values specified via Module.set_lrp_parameters(lrp_var,lrp_param)

        If lrp parameters have been pre-specified (per layer), the corresponding decomposition
        will be applied during a call of lrp().

        Specifying lrp parameters explicitly when calling lrp(), e.g. net.lrp(R,lrp_var='alpha',param=2.),
        will override the preset values for the current call.

        How to use:

        net.forward(X) #forward feed some data you wish to explain to populat the net.

        then either:

        net.lrp() #to perform the naive approach to lrp implemented in _simple_lrp for each layer

        or:

        for m in net.modules:
            m.set_lrp_parameters(...)
        net.lrp() #to preset a lrp configuration to each layer in the net

        or:

        net.lrp(somevariantname,someparameter) # to explicitly call the specified parametrization for all layers (where applicable) and override any preset configurations.

        Parameters
        ----------
        R : numpy.ndarray
            final layer relevance values. usually the network's prediction of some data points
            for which the output relevance is to be computed
            dimensionality should be equal to the previously computed predictions

        lrp_var : str
            either 'none' or 'simple' or None for standard Lrp ,
            'epsilon' for an added epsilon slack in the denominator
            'alphabeta' or 'alpha' for weighting positive and negative contributions separately. param specifies alpha with alpha + beta = 1
            'flat' projects an upper layer neuron's relevance uniformly over its receptive field.
            'ww' or 'w^2' only considers the square weights w_ij^2 as qantities to distribute relevances with.

        param : double
            the respective parameter for the lrp method of choice

        Returns
        -------

        R : numpy.ndarray
            the first layer relevances as produced by the neural net wrt to the previously forward
            passed input data. dimensionality is equal to the previously into forward entered input data

        Note
        ----

        Requires the net to be populated with temporary variables, i.e. forward needed to be called with the input
        for which the explanation is to be computed. calling clean in between forward and lrp invalidates the
        temporary data
        '''

        res_list = []
        if target_layer == None:
            for module in reversed(list(self.children())):
                res_list.append(R.detach().cpu().numpy())
                R = module.lrp(R, labels, lrp_var, param)
            return R

        else:
            for key, module in reversed(list(self._modules.items())):
                requires_activation = (key == target_layer)
                R = module.lrp(R, labels, lrp_var, param)
                if requires_activation:
                    break
            R = torch.sum(R, dim=1)
            R = torch.nn.functional.threshold(R, threshold=0, value=0)

            R_max, _ = R.max(1, keepdim=True)
            R_max, _ = R_max.max(2, keepdim=True)
            R_min, _ = R.min(1, keepdim=True)
            R_min, _ = R_min.min(2, keepdim=True)

            R = (R - R_min) / (R_max - R_min + 1e-8)

        return R


class ModuleList(Module):
    r"""Holds submodules in a list.

    ModuleList can be indexed like a regular Python list, but modules it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = operator.index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, module):
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    ModuleDict can be indexed like a regular Python dictionary, but modules it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key/value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()

    def pop(self, key):
        r"""Remove key from the ModuleDict and return its module.

        Arguments:
            key (string): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()

    def items(self):
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()

    def values(self):
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules):
        r"""Update the ModuleDict with the key/value pairs from a mapping or
        an iterable, overwriting existing keys.

        Arguments:
            modules (iterable): a mapping (dictionary) of (string: :class:`~torch.nn.Module``) or
                an iterable of key/value pairs of type (string, :class:`~torch.nn.Module``)
        """
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, Mapping):
            if isinstance(modules, OrderedDict):
                for key, module in modules.items():
                    self[key] = module
            else:
                for key, module in sorted(modules.items()):
                    self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                self[m[0]] = m[1]


class ParameterList(Module):
    r"""Holds parameters in a list.

    ParameterList can be indexed like a regular Python list, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        parameters (iterable, optional): an iterable of :class:`~torch.nn.Parameter`` to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        if parameters is not None:
            self += parameters

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ParameterList(list(self._parameters.values())[idx])
        else:
            idx = operator.index(idx)
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range'.format(idx))
            if idx < 0:
                idx += len(self)
            return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        idx = operator.index(idx)
        return self.register_parameter(str(idx), param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, parameter):
        """Appends a given parameter at the end of the list.

        Arguments:
            parameter (nn.Parameter): parameter to append
        """
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters):
        """Appends parameters from a Python iterable to the end of the list.

        Arguments:
            parameters (iterable): iterable of parameters to append
        """
        if not isinstance(parameters, Iterable):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self

    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p.data), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr


class ParameterDict(Module):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.nn.Parameter``) or an iterable of key,value pairs
            of type (string, :class:`~torch.nn.Parameter``)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters=None):
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, parameter):
        self.register_parameter(key, parameter)

    def __delitem__(self, key):
        del self._parameters[key]

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.keys())

    def __contains__(self, key):
        return key in self._parameters

    def clear(self):
        """Remove all items from the ParameterDict.
        """
        self._parameters.clear()

    def pop(self, key):
        r"""Remove key from the ParameterDict and return its parameter.

        Arguments:
            key (string): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the ParameterDict keys.
        """
        return self._parameters.keys()

    def items(self):
        r"""Return an iterable of the ParameterDict key/value pairs.
        """
        return self._parameters.items()

    def values(self):
        r"""Return an iterable of the ParameterDict values.
        """
        return self._parameters.values()

    def update(self, parameters):
        r"""Update the ParameterDict with the key/value pairs from a mapping or
        an iterable, overwriting existing keys.

        Arguments:
            parameters (iterable): a mapping (dictionary) of
                (string : :class:`~torch.nn.Parameter``) or an iterable of
                key/value pairs of type (string, :class:`~torch.nn.Parameter``)
        """
        if not isinstance(parameters, Iterable):
            raise TypeError("ParametersDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(parameters).__name__)

        if isinstance(parameters, Mapping):
            if isinstance(parameters, OrderedDict):
                for key, parameter in parameters.items():
                    self[key] = parameter
            else:
                for key, parameter in sorted(parameters.items()):
                    self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, Iterable):
                    raise TypeError("ParameterDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(p).__name__)
                if not len(p) == 2:
                    raise ValueError("ParameterDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(p)) +
                                     "; 2 is required")
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p.data), size_str, device_str)
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr
