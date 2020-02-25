import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
from PIL import Image

### MLP 모델


def relu(x):
    return np.maximum(x, 0)


def relu_derv(y):
    return np.sign(y)


def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))


def sigmoid_derv(y):
    return y * (1 - y)


def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))


def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def tanh_derv(y):
    return (1.0 + y) * (1.0 - y)


def softmax(x):
    max_elem = np.max(x, axis=1)
    diff = (x.transpose() - max_elem).transpose()
    exp = np.exp(diff)
    sum_exp = np.sum(exp, axis=1)
    probs = (exp.transpose() / sum_exp).transpose()
    return probs


def softmax_cross_entropy_with_logits(labels, logits):
    probs = softmax(logits)
    return -np.sum(labels * np.log(probs + 1.0e-10), axis=1)


def softmax_cross_entropy_with_logits_derv(labels, logits):
    return softmax(logits) - labels


def load_csv(path, skip_header=True):
    with open(path) as csvfile:
        csvreader = csv.reader(csvfile)
        headers = None
        if skip_header: headers = next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    return rows, headers


def onehot(xs, cnt):
    return np.eye(cnt)[np.array(xs).astype(int)]


def vector_to_str(x, fmt='%.2f', max_cnt=0):
    if max_cnt == 0 or len(x) <= max_cnt:
        return '[' + ','.join([fmt] * len(x)) % tuple(x) + ']'
    v = x[0:max_cnt]
    return '[' + ','.join([fmt] * len(v)) % tuple(v) + ',...]'


def load_image_pixels(imagepath, resolution, input_shape):
    img = Image.open(imagepath)
    resized = img.resize(resolution)
    return np.array(resized).reshape(input_shape)


def draw_images_horz(xs, image_shape=None):
    show_cnt = len(xs)
    fig, axes = plt.subplots(1, show_cnt, figsize=(5, 5))
    for n in range(show_cnt):
        img = xs[n]
        if image_shape:
            x3d = img.reshape(image_shape)
            img = Image.fromarray(np.uint8(x3d))
        axes[n].imshow(img)
        axes[n].axis('off')
    plt.draw()
    plt.show()


def show_select_results(est, ans, target_names, max_cnt=0):
    for n in range(len(est)):
        pstr = vector_to_str(100 * est[n], '%2.0f', max_cnt)
        estr = target_names[np.argmax(est[n])]
        astr = target_names[np.argmax(ans[n])]
        rstr = 'O'
        if estr != astr: rstr = 'X'
        print('추정확률분포 {} => 추정 {} : 정답 {} => {}'. \
              format(pstr, estr, astr, rstr))


def list_dir(path):
    filenames = os.listdir(path)
    filenames.sort()
    return filenames


# mlp model

np.random.seed(1234)


def randomize(): np.random.seed(time.time())


class Model(object):
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.is_training = False
        if not hasattr(self, 'rand_std'): self.rand_std = 0.030

    def __str__(self):
        return '{}/{}'.format(self.name, self.dataset)

    def exec_all(self, epoch_count=10, batch_size=10, learning_rate=0.001,
                 report=0, show_cnt=3):
        self.train(epoch_count, batch_size, learning_rate, report)
        self.test()
        if show_cnt > 0: self.visualize(show_cnt)


class MlpModel(Model):
    def __init__(self, name, dataset, hconfigs):
        super(MlpModel, self).__init__(name, dataset)
        self.init_parameters(hconfigs)


def mlp_init_parameters(self, hconfigs):
    self.hconfigs = hconfigs
    self.pm_hiddens = []

    prev_shape = self.dataset.input_shape

    for hconfig in hconfigs:
        pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
        self.pm_hiddens.append(pm_hidden)

    output_cnt = int(np.prod(self.dataset.output_shape))
    self.pm_output, _ = self.alloc_layer_param(prev_shape, output_cnt)


def mlp_alloc_layer_param(self, input_shape, hconfig):
    input_cnt = np.prod(input_shape)
    output_cnt = hconfig

    weight, bias = self.alloc_param_pair([input_cnt, output_cnt])

    return {'w': weight, 'b': bias}, output_cnt


def mlp_alloc_param_pair(self, shape):
    weight = np.random.normal(0, self.rand_std, shape)
    bias = np.zeros([shape[-1]])
    return weight, bias


MlpModel.init_parameters = mlp_init_parameters
MlpModel.alloc_layer_param = mlp_alloc_layer_param
MlpModel.alloc_param_pair = mlp_alloc_param_pair


def mlp_model_train(self, epoch_count=10, batch_size=10, \
                    learning_rate=0.001, report=0):
    self.learning_rate = learning_rate

    batch_count = int(self.dataset.train_count / batch_size)
    time1 = time2 = int(time.time())
    if report != 0:
        print('Model {} train started:'.format(self.name))

    for epoch in range(epoch_count):
        costs = []
        accs = []
        self.dataset.shuffle_train_data(batch_size * batch_count)
        for n in range(batch_count):
            trX, trY = self.dataset.get_train_data(batch_size, n)
            cost, acc = self.train_step(trX, trY)
            costs.append(cost)
            accs.append(acc)

        if report > 0 and (epoch + 1) % report == 0:
            vaX, vaY = self.dataset.get_validate_data(100)
            acc = self.eval_accuracy(vaX, vaY)
            time3 = int(time.time())
            tm1, tm2 = time3 - time2, time3 - time1
            self.dataset.train_prt_result(epoch + 1, costs, accs, acc, tm1, tm2)
            time2 = time3

    tm_total = int(time.time()) - time1
    print('Model {} train ended in {} secs:'.format(self.name, tm_total))


MlpModel.train = mlp_model_train


def mlp_model_test(self):
    teX, teY = self.dataset.get_test_data()
    time1 = int(time.time())
    acc = self.eval_accuracy(teX, teY)
    time2 = int(time.time())
    self.dataset.test_prt_result(self.name, acc, time2 - time1)


MlpModel.test = mlp_model_test


def mlp_model_visualize(self, num):
    print('Model {} Visualization'.format(self.name))
    deX, deY = self.dataset.get_visualize_data(num)
    est = self.get_estimate(deX)
    self.dataset.visualize(deX, est, deY)


MlpModel.visualize = mlp_model_visualize


def mlp_train_step(self, x, y):
    self.is_training = True

    output, aux_nn = self.forward_neuralnet(x)
    loss, aux_pp = self.forward_postproc(output, y)
    accuracy = self.eval_accuracy(x, y, output)

    G_loss = 1.0
    G_output = self.backprop_postproc(G_loss, aux_pp)
    self.backprop_neuralnet(G_output, aux_nn)

    self.is_training = False

    return loss, accuracy


MlpModel.train_step = mlp_train_step


def mlp_forward_neuralnet(self, x):
    hidden = x
    aux_layers = []

    for n, hconfig in enumerate(self.hconfigs):
        hidden, aux = self.forward_layer(hidden, hconfig, self.pm_hiddens[n])
        aux_layers.append(aux)

    output, aux_out = self.forward_layer(hidden, None, self.pm_output)

    return output, [aux_out, aux_layers]


def mlp_backprop_neuralnet(self, G_output, aux):
    aux_out, aux_layers = aux

    G_hidden = self.backprop_layer(G_output, None, self.pm_output, aux_out)

    for n in reversed(range(len(self.hconfigs))):
        hconfig, pm, aux = self.hconfigs[n], self.pm_hiddens[n], aux_layers[n]
        G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)

    return G_hidden


MlpModel.forward_neuralnet = mlp_forward_neuralnet
MlpModel.backprop_neuralnet = mlp_backprop_neuralnet


def mlp_forward_layer(self, x, hconfig, pm):
    y = np.matmul(x, pm['w']) + pm['b']
    if hconfig is not None: y = relu(y)
    return y, [x, y]


def mlp_backprop_layer(self, G_y, hconfig, pm, aux):
    x, y = aux

    if hconfig is not None: G_y = relu_derv(y) * G_y

    g_y_weight = x.transpose()
    g_y_input = pm['w'].transpose()

    G_weight = np.matmul(g_y_weight, G_y)
    G_bias = np.sum(G_y, axis=0)
    G_input = np.matmul(G_y, g_y_input)

    pm['w'] -= self.learning_rate * G_weight
    pm['b'] -= self.learning_rate * G_bias

    return G_input


MlpModel.forward_layer = mlp_forward_layer
MlpModel.backprop_layer = mlp_backprop_layer


def mlp_forward_postproc(self, output, y):
    loss, aux_loss = self.dataset.forward_postproc(output, y)
    extra, aux_extra = self.forward_extra_cost(y)
    return loss + extra, [aux_loss, aux_extra]


def mlp_forward_extra_cost(self, y):
    return 0, None


MlpModel.forward_postproc = mlp_forward_postproc
MlpModel.forward_extra_cost = mlp_forward_extra_cost


def mlp_backprop_postproc(self, G_loss, aux):
    aux_loss, aux_extra = aux
    self.backprop_extra_cost(G_loss, aux_extra)
    G_output = self.dataset.backprop_postproc(G_loss, aux_loss)
    return G_output


def mlp_backprop_extra_cost(self, G_loss, aux):
    pass


MlpModel.backprop_postproc = mlp_backprop_postproc
MlpModel.backprop_extra_cost = mlp_backprop_extra_cost


def mlp_eval_accuracy(self, x, y, output=None):
    if output is None:
        output, _ = self.forward_neuralnet(x)
    accuracy = self.dataset.eval_accuracy(x, y, output)
    return accuracy


MlpModel.eval_accuracy = mlp_eval_accuracy


def mlp_get_estimate(self, x):
    output, _ = self.forward_neuralnet(x)
    estimate = self.dataset.get_estimate(output)
    return estimate


MlpModel.get_estimate = mlp_get_estimate


# dataset_flowers

class Dataset(object):
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode

    def __str__(self):
        return '{}({}, {}+{}+{})'.format(self.name, self.mode, \
                                         len(self.tr_xs), len(self.te_xs), len(self.va_xs))

    @property
    def train_count(self):
        return len(self.tr_xs)


def dataset_get_train_data(self, batch_size, nth):
    from_idx = nth * batch_size
    to_idx = (nth + 1) * batch_size

    tr_X = self.tr_xs[self.indices[from_idx:to_idx]]
    tr_Y = self.tr_ys[self.indices[from_idx:to_idx]]

    return tr_X, tr_Y


def dataset_shuffle_train_data(self, size):
    self.indices = np.arange(size)
    np.random.shuffle(self.indices)


Dataset.get_train_data = dataset_get_train_data
Dataset.shuffle_train_data = dataset_shuffle_train_data


def dataset_get_test_data(self):
    return self.te_xs, self.te_ys


Dataset.get_test_data = dataset_get_test_data


def dataset_get_validate_data(self, count):
    self.va_indices = np.arange(len(self.va_xs))
    np.random.shuffle(self.va_indices)

    va_X = self.va_xs[self.va_indices[0:count]]
    va_Y = self.va_ys[self.va_indices[0:count]]

    return va_X, va_Y


Dataset.get_validate_data = dataset_get_validate_data
Dataset.get_visualize_data = dataset_get_validate_data


def dataset_shuffle_data(self, xs, ys, tr_ratio=0.8, va_ratio=0.05):
    data_count = len(xs)

    tr_cnt = int(data_count * tr_ratio / 10) * 10
    va_cnt = int(data_count * va_ratio)
    te_cnt = data_count - (tr_cnt + va_cnt)

    tr_from, tr_to = 0, tr_cnt
    va_from, va_to = tr_cnt, tr_cnt + va_cnt
    te_from, te_to = tr_cnt + va_cnt, data_count

    indices = np.arange(data_count)
    np.random.shuffle(indices)

    self.tr_xs = xs[indices[tr_from:tr_to]]
    self.tr_ys = ys[indices[tr_from:tr_to]]
    self.va_xs = xs[indices[va_from:va_to]]
    self.va_ys = ys[indices[va_from:va_to]]
    self.te_xs = xs[indices[te_from:te_to]]
    self.te_ys = ys[indices[te_from:te_to]]

    self.input_shape = xs[0].shape
    self.output_shape = ys[0].shape

    return indices[tr_from:tr_to], indices[va_from:va_to], indices[te_from:te_to]


Dataset.shuffle_data = dataset_shuffle_data


def dataset_forward_postproc(self, output, y, mode=None):
    if mode is None: mode = self.mode

    if mode == 'regression':
        diff = output - y
        square = np.square(diff)
        loss = np.mean(square)
        aux = diff
    elif mode == 'binary':
        entropy = sigmoid_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        aux = [y, output]
    elif mode == 'select':
        entropy = softmax_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        aux = [output, y, entropy]

    return loss, aux


Dataset.forward_postproc = dataset_forward_postproc


def dataset_backprop_postproc(self, G_loss, aux, mode=None):
    if mode is None: mode = self.mode

    if mode == 'regression':
        diff = aux
        shape = diff.shape

        g_loss_square = np.ones(shape) / np.prod(shape)
        g_square_diff = 2 * diff
        g_diff_output = 1

        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        G_output = g_diff_output * G_diff
    elif mode == 'binary':
        y, output = aux
        shape = output.shape

        g_loss_entropy = np.ones(shape) / np.prod(shape)
        g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy
    elif mode == 'select':
        output, y, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = softmax_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

    return G_output


Dataset.backprop_postproc = dataset_backprop_postproc


def dataset_eval_accuracy(self, x, y, output, mode=None):
    if mode is None: mode = self.mode

    if mode == 'regression':
        mse = np.mean(np.square(output - y))
        accuracy = 1 - np.sqrt(mse) / np.mean(y)
    elif mode == 'binary':
        estimate = np.greater(output, 0)
        answer = np.equal(y, 1.0)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)
    elif mode == 'select':
        estimate = np.argmax(output, axis=1)
        answer = np.argmax(y, axis=1)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)

    return accuracy


Dataset.eval_accuracy = dataset_eval_accuracy


def dataset_get_estimate(self, output, mode=None):
    if mode is None: mode = self.mode

    if mode == 'regression':
        estimate = output
    elif mode == 'binary':
        estimate = sigmoid(output)
    elif mode == 'select':
        estimate = softmax(output)

    return estimate


Dataset.get_estimate = dataset_get_estimate


def dataset_train_prt_result(self, epoch, costs, accs, acc, time1, time2):
    print('    Epoch {}: cost={:5.3f}, accuracy={:5.3f}/{:5.3f} ({}/{} secs)'. \
          format(epoch, np.mean(costs), np.mean(accs), acc, time1, time2))


def dataset_test_prt_result(self, name, acc, time):
    print('Model {} test report: accuracy = {:5.3f}, ({} secs)\n'. \
          format(name, acc, time))


Dataset.train_prt_result = dataset_train_prt_result
Dataset.test_prt_result = dataset_test_prt_result


class FlowersDataset(Dataset):
    pass


def flowers_init(self, resolution=[100, 100], input_shape=[-1]):
    super(FlowersDataset, self).__init__('flowers', 'select')

    path = 'flowers'
    self.target_names = list_dir(path)

    images = []
    idxs = []

    for dx, dname in enumerate(self.target_names):
        subpath = path + '/' + dname
        filenames = list_dir(subpath)
        for fname in filenames:
            if fname[-4:] != '.jpg':
                continue
            imagepath = os.path.join(subpath, fname)
            pixels = load_image_pixels(imagepath, resolution, input_shape)
            images.append(pixels)
            idxs.append(dx)

    self.image_shape = resolution + [3]

    xs = np.asarray(images, np.float32)
    ys = onehot(idxs, len(self.target_names))

    self.shuffle_data(xs, ys, 0.8)


FlowersDataset.__init__ = flowers_init


def flowers_visualize(self, xs, estimates, answers):
    draw_images_horz(xs, self.image_shape)
    show_select_results(estimates, answers, self.target_names)


FlowersDataset.visualize = flowers_visualize



#adam 알고리즘
class AdamModel(MlpModel):
    def __init__(self, name, dataset, hconfigs):
        self.use_adam = False
        super(AdamModel, self).__init__(name, dataset, hconfigs)


def adam_backprop_layer(self, G_y, hconfig, pm, aux):
    x, y = aux

    if hconfig is not None: G_y = relu_derv(y) * G_y

    g_y_weight = x.transpose()
    g_y_input = pm['w'].transpose()

    G_weight = np.matmul(g_y_weight, G_y)
    G_bias = np.sum(G_y, axis=0)
    G_input = np.matmul(G_y, g_y_input)

    self.update_param(pm, 'w', G_weight)
    self.update_param(pm, 'b', G_bias)

    return G_input


AdamModel.backprop_layer = adam_backprop_layer


def adam_update_param(self, pm, key, delta):
    if self.use_adam:
        delta = self.eval_adam_delta(pm, key, delta)

    pm[key] -= self.learning_rate * delta


AdamModel.update_param = adam_update_param


def adam_eval_adam_delta(self, pm, key, delta):
    ro_1 = 0.9
    ro_2 = 0.999
    epsilon = 1.0e-8

    skey, tkey, step = 's' + key, 't' + key, 'n' + key
    if skey not in pm:
        pm[skey] = np.zeros(pm[key].shape)
        pm[tkey] = np.zeros(pm[key].shape)
        pm[step] = 0

    s = pm[skey] = ro_1 * pm[skey] + (1 - ro_1) * delta
    t = pm[tkey] = ro_2 * pm[tkey] + (1 - ro_2) * (delta * delta)

    pm[step] += 1
    s = s / (1 - np.power(ro_1, pm[step]))
    t = t / (1 - np.power(ro_2, pm[step]))

    return s / (np.sqrt(t) + epsilon)


AdamModel.eval_adam_delta = adam_eval_adam_delta


###CNN 모델 설계
class CnnBasicModel(AdamModel):
    def __init__(self, name, dataset, hconfigs, show_maps = False):
        if isinstance(hconfigs, list) and \
        not isinstance(hconfigs[0], (list, int)):
            hconfigs = [hconfigs]
        self.show_maps = show_maps
        self.need_maps = False
        self.kernels = []
        super(CnnBasicModel, self).__init__(name, dataset, hconfigs)
        self.use_adam = True


def cnn_basic_alloc_layer_param(self, input_shape, hconfig):
    layer_type = get_layer_type(hconfig)

    m_name = 'alloc_{}_layer'.format(layer_type)
    method = getattr(self, m_name)
    pm, output_shape = method(input_shape, hconfig)

    return pm, output_shape


CnnBasicModel.alloc_layer_param = cnn_basic_alloc_layer_param


def cnn_basic_forward_layer(self, x, hconfig, pm):
    layer_type = get_layer_type(hconfig)

    m_name = 'forward_{}_layer'.format(layer_type)
    method = getattr(self, m_name)
    y, aux = method(x, hconfig, pm)

    return y, aux


CnnBasicModel.forward_layer = cnn_basic_forward_layer


def cnn_basic_backprop_layer(self, G_y, hconfig, pm, aux):
    layer_type = get_layer_type(hconfig)

    m_name = 'backprop_{}_layer'.format(layer_type)
    method = getattr(self, m_name)
    G_input = method(G_y, hconfig, pm, aux)

    return G_input


CnnBasicModel.backprop_layer = cnn_basic_backprop_layer

def cnn_basic_alloc_full_layer(self, input_shape, hconfig):
    input_cnt = np.prod(input_shape)
    output_cnt = get_conf_param(hconfig, 'width', hconfig)

    weight = np.random.normal(0, self.rand_std, [input_cnt, output_cnt])
    bias = np.zeros([output_cnt])

    return {'w':weight, 'b':bias}, [output_cnt]


def cnn_basic_alloc_conv_layer(self, input_shape, hconfig):
    assert len(input_shape) == 3
    xh, xw, xchn = input_shape
    kh, kw = get_conf_param_2d(hconfig, 'ksize')
    ychn = get_conf_param(hconfig, 'chn')

    kernel = np.random.normal(0, self.rand_std, [kh, kw, xchn, ychn])
    bias = np.zeros([ychn])

    if self.show_maps: self.kernels.append(kernel)

    return {'k': kernel, 'b': bias}, [xh, xw, ychn]


def cnn_basic_alloc_pool_layer(self, input_shape, hconfig):
    assert len(input_shape) == 3
    xh, xw, xchn = input_shape
    sh, sw = get_conf_param_2d(hconfig, 'stride')

    assert xh % sh == 0
    assert xw % sw == 0

    return {}, [xh // sh, xw // sw, xchn]


CnnBasicModel.alloc_full_layer = cnn_basic_alloc_full_layer
CnnBasicModel.alloc_conv_layer = cnn_basic_alloc_conv_layer
CnnBasicModel.alloc_max_layer = cnn_basic_alloc_pool_layer
CnnBasicModel.alloc_avg_layer = cnn_basic_alloc_pool_layer


def get_layer_type(hconfig):
    if not isinstance(hconfig, list): return 'full'
    return hconfig[0]


def get_conf_param(hconfig, key, defval=None):
    if not isinstance(hconfig, list): return defval
    if len(hconfig) <= 1: return defval
    if not key in hconfig[1]: return defval
    return hconfig[1][key]


def get_conf_param_2d(hconfig, key, defval=None):
    if len(hconfig) <= 1: return defval
    if not key in hconfig[1]: return defval
    val = hconfig[1][key]
    if isinstance(val, list): return val
    return [val, val]


def cnn_basic_forward_full_layer(self, x, hconfig, pm):
    if pm is None: return x, None

    x_org_shape = x.shape

    if len(x.shape) != 2:
        mb_size = x.shape[0]
        x = x.reshape([mb_size, -1])

    affine = np.matmul(x, pm['w']) + pm['b']
    y = self.activate(affine, hconfig)

    return y, [x, y, x_org_shape]


CnnBasicModel.forward_full_layer = cnn_basic_forward_full_layer


def cnn_basic_backprop_full_layer(self, G_y, hconfig, pm, aux):
    if pm is None: return G_y

    x, y, x_org_shape = aux

    G_affine = self.activate_derv(G_y, y, hconfig)

    g_affine_weight = x.transpose()
    g_affine_input = pm['w'].transpose()

    G_weight = np.matmul(g_affine_weight, G_affine)
    G_bias = np.sum(G_affine, axis=0)
    G_input = np.matmul(G_affine, g_affine_input)

    self.update_param(pm, 'w', G_weight)
    self.update_param(pm, 'b', G_bias)

    return G_input.reshape(x_org_shape)


CnnBasicModel.backprop_full_layer = cnn_basic_backprop_full_layer


def cnn_basic_activate(self, affine, hconfig):
    if hconfig is None: return affine

    func = get_conf_param(hconfig, 'actfunc', 'relu')

    if func == 'none':
        return affine
    elif func == 'relu':
        return relu(affine)
    elif func == 'sigmoid':
        return sigmoid(affine)
    elif func == 'tanh':
        return tanh(affine)
    else:
        assert 0


def cnn_basic_activate_derv(self, G_y, y, hconfig):
    if hconfig is None: return G_y

    func = get_conf_param(hconfig, 'actfunc', 'relu')

    if func == 'none':
        return G_y
    elif func == 'relu':
        return relu_derv(y) * G_y
    elif func == 'sigmoid':
        return sigmoid_derv(y) * G_y
    elif func == 'tanh':
        return tanh_derv(y) * G_y
    else:
        assert 0


CnnBasicModel.activate = cnn_basic_activate
CnnBasicModel.activate_derv = cnn_basic_activate_derv


def forward_conv_layer_adhoc(self, x, hconfig, pm):
    mb_size, xh, xw, xchn = x.shape
    kh, kw, _, ychn = pm['k'].shape

    conv = np.zeros((mb_size, xh, xw, ychn))

    for n in range(mb_size):
        for r in range(xh):
            for c in range(xw):
                for ym in range(ychn):
                    for i in range(kh):
                        for j in range(kw):
                            rx = r + i - (kh - 1) // 2
                            cx = c + j - (kw - 1) // 2
                            if rx < 0 or rx >= xh: continue
                            if cx < 0 or cx >= xw: continue
                            for xm in range(xchn):
                                kval = pm['k'][i][j][xm][ym]
                                ival = x[n][rx][cx][xm]
                                conv[n][r][c][ym] += kval * ival

    y = self.activate(conv + pm['b'], hconfig)

    return y, [x, y]


def forward_conv_layer_better(self, x, hconfig, pm):
    mb_size, xh, xw, xchn = x.shape
    kh, kw, _, ychn = pm['k'].shape

    conv = np.zeros((mb_size, xh, xw, ychn))

    bh, bw = (kh - 1) // 2, (kw - 1) // 2
    eh, ew = xh + kh - 1, xw + kw - 1

    x_ext = np.zeros((mb_size, eh, ew, xchn))
    x_ext[:, bh:bh + xh, bw:bw + xw, :] = x

    k_flat = pm['k'].transpose([3, 0, 1, 2]).reshape([ychn, -1])

    for n in range(mb_size):
        for r in range(xh):
            for c in range(xw):
                for ym in range(ychn):
                    xe_flat = x_ext[n, r:r + kh, c:c + kw, :].flatten()
                    conv[n, r, c, ym] = (xe_flat * k_flat[ym]).sum()

    y = self.activate(conv + pm['b'], hconfig)

    return y, [x, y]


def cnn_basic_forward_conv_layer(self, x, hconfig, pm):
    mb_size, xh, xw, xchn = x.shape
    kh, kw, _, ychn = pm['k'].shape

    x_flat = get_ext_regions_for_conv(x, kh, kw)
    k_flat = pm['k'].reshape([kh * kw * xchn, ychn])
    conv_flat = np.matmul(x_flat, k_flat)
    conv = conv_flat.reshape([mb_size, xh, xw, ychn])

    y = self.activate(conv + pm['b'], hconfig)

    if self.need_maps: self.maps.append(y)

    return y, [x_flat, k_flat, x, y]


CnnBasicModel.forward_conv_layer = cnn_basic_forward_conv_layer


def cnn_basic_backprop_conv_layer(self, G_y, hconfig, pm, aux):
    x_flat, k_flat, x, y = aux

    kh, kw, xchn, ychn = pm['k'].shape
    mb_size, xh, xw, _ = G_y.shape

    G_conv = self.activate_derv(G_y, y, hconfig)

    G_conv_flat = G_conv.reshape(mb_size * xh * xw, ychn)

    g_conv_k_flat = x_flat.transpose()
    g_conv_x_flat = k_flat.transpose()

    G_k_flat = np.matmul(g_conv_k_flat, G_conv_flat)
    G_x_flat = np.matmul(G_conv_flat, g_conv_x_flat)
    G_bias = np.sum(G_conv_flat, axis=0)

    G_kernel = G_k_flat.reshape([kh, kw, xchn, ychn])
    G_input = undo_ext_regions_for_conv(G_x_flat, x, kh, kw)

    self.update_param(pm, 'k', G_kernel)
    self.update_param(pm, 'b', G_bias)

    return G_input


CnnBasicModel.backprop_conv_layer = cnn_basic_backprop_conv_layer


def get_ext_regions_for_conv(x, kh, kw):
    mb_size, xh, xw, xchn = x.shape

    regs = get_ext_regions(x, kh, kw, 0)
    regs = regs.transpose([2, 0, 1, 3, 4, 5])

    return regs.reshape([mb_size * xh * xw, kh * kw * xchn])


def get_ext_regions(x, kh, kw, fill):
    mb_size, xh, xw, xchn = x.shape

    eh, ew = xh + kh - 1, xw + kw - 1
    bh, bw = (kh - 1) // 2, (kw - 1) // 2

    x_ext = np.zeros((mb_size, eh, ew, xchn), dtype='float32') + fill
    x_ext[:, bh:bh + xh, bw:bw + xw, :] = x

    regs = np.zeros((xh, xw, mb_size * kh * kw * xchn), dtype='float32')

    for r in range(xh):
        for c in range(xw):
            regs[r, c, :] = x_ext[:, r:r + kh, c:c + kw, :].flatten()

    return regs.reshape([xh, xw, mb_size, kh, kw, xchn])


def undo_ext_regions_for_conv(regs, x, kh, kw):
    mb_size, xh, xw, xchn = x.shape

    regs = regs.reshape([mb_size, xh, xw, kh, kw, xchn])
    regs = regs.transpose([1, 2, 0, 3, 4, 5])

    return undo_ext_regions(regs, kh, kw)


def undo_ext_regions(regs, kh, kw):
    xh, xw, mb_size, kh, kw, xchn = regs.shape

    eh, ew = xh + kh - 1, xw + kw - 1
    bh, bw = (kh - 1) // 2, (kw - 1) // 2

    gx_ext = np.zeros([mb_size, eh, ew, xchn], dtype='float32')

    for r in range(xh):
        for c in range(xw):
            gx_ext[:, r:r + kh, c:c + kw, :] += regs[r, c]

    return gx_ext[:, bh:bh + xh, bw:bw + xw, :]


def cnn_basic_forward_avg_layer(self, x, hconfig, pm):
    mb_size, xh, xw, chn = x.shape
    sh, sw = get_conf_param_2d(hconfig, 'stride')
    yh, yw = xh // sh, xw // sw

    x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
    x2 = x1.transpose(0, 1, 3, 5, 2, 4)
    x3 = x2.reshape([-1, sh * sw])

    y_flat = np.average(x3, 1)
    y = y_flat.reshape([mb_size, yh, yw, chn])

    if self.need_maps: self.maps.append(y)

    return y, None


def cnn_basic_backprop_avg_layer(self, G_y, hconfig, pm, aux):
    mb_size, yh, yw, chn = G_y.shape
    sh, sw = get_conf_param_2d(hconfig, 'stride')
    xh, xw = yh * sh, yw * sw

    gy_flat = G_y.flatten() / (sh * sw)

    gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
    for i in range(sh * sw):
        gx1[:, i] = gy_flat
    gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
    gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

    G_input = gx3.reshape([mb_size, xh, xw, chn])

    return G_input


CnnBasicModel.forward_avg_layer = cnn_basic_forward_avg_layer
CnnBasicModel.backprop_avg_layer = cnn_basic_backprop_avg_layer


def cnn_basic_forward_max_layer(self, x, hconfig, pm):
    mb_size, xh, xw, chn = x.shape
    sh, sw = get_conf_param_2d(hconfig, 'stride')
    yh, yw = xh // sh, xw // sw

    x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
    x2 = x1.transpose(0, 1, 3, 5, 2, 4)
    x3 = x2.reshape([-1, sh * sw])

    idxs = np.argmax(x3, axis=1)
    y_flat = x3[np.arange(mb_size * yh * yw * chn), idxs]
    y = y_flat.reshape([mb_size, yh, yw, chn])

    if self.need_maps: self.maps.append(y)

    return y, idxs


def cnn_basic_backprop_max_layer(self, G_y, hconfig, pm, aux):
    idxs = aux

    mb_size, yh, yw, chn = G_y.shape
    sh, sw = get_conf_param_2d(hconfig, 'stride')
    xh, xw = yh * sh, yw * sw

    gy_flat = G_y.flatten()

    gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
    gx1[np.arange(mb_size * yh * yw * chn), idxs] = gy_flat[:]
    gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
    gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

    G_input = gx3.reshape([mb_size, xh, xw, chn])

    return G_input


CnnBasicModel.forward_max_layer = cnn_basic_forward_max_layer
CnnBasicModel.backprop_max_layer = cnn_basic_backprop_max_layer


def cnn_basic_visualize(self, num):
    print('Model {} Visualization'.format(self.name))

    self.need_maps = self.show_maps
    self.maps = []

    deX, deY = self.dataset.get_visualize_data(num)
    est = self.get_estimate(deX)

    if self.show_maps:
        for kernel in self.kernels:
            kh, kw, xchn, ychn = kernel.shape
            grids = kernel.reshape([kh, kw, -1]).transpose(2, 0, 1)
            draw_images_horz(grids[0:5, :, :])

        for pmap in self.maps:
            draw_images_horz(pmap[:, :, :, 0])

    self.dataset.visualize(deX, est, deY)

    self.need_maps = False
    self.maps = None


CnnBasicModel.visualize = cnn_basic_visualize

#main
fd = FlowersDataset([96, 96], [96, 96, 3])

'''
# 다층 퍼셉트론 처리
fm1 = CnnBasicModel('flowers_model_1', fd, [30, 10])
fm1.exec_all(epoch_count = 10, report = 2)

#완전연결계층의 은닉 계층 구성
fm2 = CnnBasicModel('flowers_model_2', fd, 
                    [['full', {'width':30}], 
                     ['full', {'width':10}]])
fm2.use_adam = False
fm2.exec_all(epoch_count = 10, report = 2)


#CNN 기반 이미지처리
fm3 = CnnBasicModel('flowers_model_3', fd, 
               [['conv', {'ksize':5, 'chn':6}], 
                ['max', {'stride':4}], 
                ['conv', {'ksize':3, 'chn':12}], 
                ['avg', {'stride':2}]], 
               True)
fm3.exec_all(epoch_count = 10, report = 2)

#더 많은 계층을 갖는 합성곱신경망
fm4 = CnnBasicModel('flowers_model_4', fd, 
               [['conv', {'ksize':3, 'chn':6}], 
                ['max', {'stride':2}], 
                ['conv', {'ksize':3, 'chn':12}], 
                ['max', {'stride':2}], 
                ['conv', {'ksize':3, 'chn':24}], 
                ['avg', {'stride':3}]])
fm4.exec_all(epoch_count = 10, report = 2)
'''
