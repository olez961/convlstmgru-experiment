import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, activation=F.tanh, peephole=False, batchnorm=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        # 输入张量尺寸（不包括最后一个维度）
        self.height, self.width = input_size
        # 输入特征的维度，即输入张量最后一维的大小
        self.input_dim          = input_dim
        # 隐藏状态的维度，也即cell输出的大小
        self.hidden_dim         = hidden_dim

        # 卷积核尺寸，一般使用3*3或者5*5尺寸的卷积核
        self.kernel_size = kernel_size
        # 卷积层的填充大小，它是一个二元组 (padding_height, padding_width)，
        # 这里设置为使得输出大小和输入大小一致的填充
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        # 是否使用bias
        self.bias        = bias
        # LSTMCell 中使用的非线性激活函数，一般选用 tanh 或者 relu
        self.activation  = activation
        # 是否使用 peephole 连接，即将 cell state 引入到 input gate，output gate 中
        self.peephole    = peephole
        # 是否进行批归一化
        self.batchnorm   = batchnorm

        if peephole:
            # 输入门的peephole连接权重
            self.Wci = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))
            # 遗忘门的peephole连接权重
            self.Wcf = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))
            # 输出门的peephole连接权重
            self.Wco = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))

        # 核心卷积模块，输出四倍隐藏层大小的特征图
        # 其中包含输入门、遗忘门、输出门以及细胞状态的信息
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        # 对LSTMCell的参数进行初始化，以帮助模型更好地学习数据的表示。
        self.reset_parameters()

    # 定义LSTMCell的前向传播函数
    # input表示当前时刻的输入数据，prev_state表示上一个时刻的隐藏状态和记忆状态
    def forward(self, input, prev_state):
        # 将上一时刻的隐藏状态 $h_{t-1}$ 和细胞状态 $c_{t-1}$ 
        # 从上一个时间步的状态 prev_state 中解包出来，以便在当前时间步中使用
        # 本函数最后返回的似乎也是一个prev_state结构的数据
        h_prev, c_prev = prev_state

        # 将当前输入和前一个时间步的隐藏状态在通道维度上拼接在一起，
        # 形成一个新的张量，用于输入到卷积层中进行特征提取
        # input表示当前时间步的输入张量，
        # h_prev表示前一个时间步的隐藏状态张量，
        # dim=1表示在通道维度上进行拼接
        combined = torch.cat((input, h_prev), dim=1)  # concatenate along channel axis
        # 将输入和前一时刻的隐藏状态沿着通道维度拼接在一起，
        # 然后将拼接后的张量送入卷积层计算
        # combined是将 input 和 h_prev 沿着通道维度拼接在一起，
        # 得到一个形状为 (batch_size, input_dim+hidden_dim, height, width) 的张量。
        # 然后，将这个张量送入 self.conv 中，
        # 通过卷积计算得到一个形状为 (batch_size, 4*hidden_dim, height, width) 的张量 
        # combined_conv，
        # 其中4表示4个门（输入门，遗忘门，输出门，细胞状态门）需要计算
        combined_conv = self.conv(combined)

        # 使用torch.split函数将combined_conv按照通道维度(dim=1)分成了四份，
        # 分别对应输入门(i), 遗忘门(f), 输出门(o)以及单元状态(g)的卷积结果。
        # 这些分割出来的结果被赋值给cc_i, cc_f, cc_o, cc_g四个变量。
        # 这是因为在LSTM模型中，输入和状态被同时输入到LSTM单元中进行计算，
        # 而这四份卷积结果恰好对应着LSTM单元中需要计算的四个部分，
        # 即输入门、遗忘门、输出门以及单元状态
        # 第二个参数是什么意思是指按照self.hidden_dim这个大小来分割原张量
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 是否启用peephole来计算输入门（i）和遗忘门（f）的输出值
        if self.peephole:
            i = F.sigmoid(cc_i + self.Wci * c_prev)
            f = F.sigmoid(cc_f + self.Wcf * c_prev)
        else:
            i = F.sigmoid(cc_i)
            f = F.sigmoid(cc_f)

        # 将输入门、遗忘门、输出门和细胞状态的卷积结果中的细胞状态更新值（通常为 cc_g）
        # 经过激活函数处理得到当前时刻的候选细胞状态值 g。
        # 常见的激活函数包括 tanh 和 ReLU，
        # 它们的作用是将输入映射到一定的范围内，增加模型的表达能力
        g = self.activation(cc_g)
        # 计算当前时刻的细胞状态
        # 可以看到细胞当前状态主要由遗忘门、输入门以及之前细胞状态和当前细胞状态决定 
        c_cur = f * c_prev + i * g

        # 根据是否启用了peephole来控制当前输出门的计算
        if self.peephole:
            # cc_o是当前输出门的卷积结果
            # self.Wco是输出门对当前cell状态c_cur的权重矩阵
            o = F.sigmoid(cc_o + self.Wco * c_cur)
        else:
            o = F.sigmoid(cc_o)

        # 计算当前时刻的隐藏层状态
        h_cur = o * self.activation(c_cur)

        # 返回当前层的隐藏层结果及细胞状态,
        # 可以看到这里的返回值同样可以是这个函数的输入值的一部分
        return h_cur, c_cur

    # 初始化LSTM模型的隐藏状态
    # 该函数返回一个元组，包含两个张量，
    # 分别代表LSTM模型的cell state和hidden state。其中，
    # cell state和hidden state的维度都为(batch_size, hidden_dim, height, width)，
    # 其中height和width分别对应输入图像的高和宽。
    # 如果使用CUDA，返回的张量会被移动到指定的CUDA设备上
    def init_hidden(self, batch_size, cuda=True, device='cuda'):
        state = (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                 torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        if cuda:
            state = (state[0].to(device), state[1].to(device))
        return state

    def reset_parameters(self):
        #self.conv.reset_parameters()
        # 对卷积层的权重进行Xavier初始化，使其具有合理的初始值，从而有助于模型的训练和收敛。
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain('tanh'))
        # 将卷积层的偏置初始化为0，使模型的初始偏置为零。
        self.conv.bias.data.zero_()

        if self.batchnorm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()
        if self.peephole:
            # std = 1. / math.sqrt(self.hidden_dim) 表示了一种初始化方法，
            # 即将权重初始化到均值为 0，标准差为 1/sqrt(self.hidden_dim) 的正态分布中
            std = 1. / math.sqrt(self.hidden_dim)
            # peephole选项开启后
            # 将这些权重随机初始化到 [0, 1] 的范围内
            self.Wci.data.uniform_(0,1)#(std=std)
            self.Wcf.data.uniform_(0,1)#(std=std)
            self.Wco.data.uniform_(0,1)#(std=std)
            # 若想使用std配置分布,应这样写:
            # torch.nn.init.normal_(self.Wci, mean=0, std=std)
            # torch.nn.init.normal_(self.Wcf, mean=0, std=std)
            # torch.nn.init.normal_(self.Wco, mean=0, std=std)


# 定义一个由多个ConvLSTMCell组成的多层ConvLSTM模型，并提供了forward函数用于模型的前向传播
class ConvLSTM(nn.Module):
    #ConvLSTM类的构造函数。它接受以下参数：
    # input_size：输入数据的空间大小（height, width）
    # input_dim：输入数据的通道数
    # hidden_dim：隐藏状态的通道数
    # kernel_size：卷积核的大小
    # num_layers：LSTM的层数
    # batch_first：输入数据的第一是否维是batch size，默认为False
    # bias：是否添加偏置，默认为True
    # activation：激活函数，默认为tanh
    # peephole：是否使用peephole连接，默认为False
    # batchnorm：是否使用批标准化，默认为False
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, activation=F.tanh, peephole=False, batchnorm=False):
        # 下面这一句一定需要吗?是的
        # 在 nn.Module 的子类中，如果不显式调用父类的 __init__ 方法，
        # 则无法正确初始化子类实例变量并继承 nn.Module 的方法。
        # 因此这一句非常重要，不能省略
        super(ConvLSTM, self).__init__()

        # 检查kernel_size参数是否合法。因为卷积核的大小需要在所有维度上是整数，
        # 并且在输入大小和输出大小之间具有一定的关系。
        # 如果kernel_size不合法，则会抛出ValueError异常
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # 检查并扩展 kernel_size 和 hidden_dim 使其满足多层LSTM的要求，
        # 即 kernel_size 和 hidden_dim 都应该是长度为 num_layers 的列表。
        # 如果 kernel_size 和 hidden_dim 长度不足 num_layers，
        # 则将它们扩展到 num_layers 的长度。
        # 如果 activation 是单一的函数，它也将被扩展成一个列表
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        activation  = self._extend_for_multilayer(activation, num_layers)

        # 检查传入参数 kernel_size, hidden_dim, activation 的长度是否与 num_layers 相等，
        # 如果不相等则会抛出一个 ValueError 异常。
        # 因为在 ConvLSTM 中的每一层中，这三个参数的长度应该相等，
        # 代表了每一层的卷积核大小，隐藏状态的维度和激活函数。
        # 如果不相等，那么就无法正确构建多层的 ConvLSTM 模型
        if not len(kernel_size) == len(hidden_dim) == len(activation) == num_layers:
            raise ValueError('Inconsistent list length.')

        # 输入数据的空间大小(高度和宽度)
        self.height, self.width = input_size

        # 输入数据的通道数
        self.input_dim  = input_dim
        # 每个LSTM 单元的隐藏状态的通道数
        self.hidden_dim = hidden_dim
        # LSTM 单元的卷积核大小，可以是一个整数或一个长度为2的元组/列表
        self.kernel_size = kernel_size
        # LSTM 单元(cell)的堆叠层数
        self.num_layers = num_layers
        # 是否将批次维放在输入张量的最前面
        self.batch_first = batch_first
        # 是否启用偏置
        self.bias = bias

        # 构建多层的ConvLSTMCell，并将其加入到cell_list中
        cell_list = []
        # 对于第一层，输入维度是input_dim，
        # 后续每一层的输入维度都是前一层的输出维度（即self.hidden_dim[i-1]）。
        # 最后，cell_list列表中保存的就是所有ConvLSTMCell
        for i in range(0, self.num_layers):
            # 这里的隐藏层真的会发生变化吗,或者它就一直是同一个值吗
            # 在这段代码中，隐藏层维度是会发生变化的。
            # 具体来说，在循环中的第i次迭代中，
            # 我们根据当前的i来决定输入维度是input_dim还是上一层的hidden_dim[i-1]，
            # 然后创建一个新的ConvLSTMCell，将其添加到cell_list中
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            # 把ConvLSTMCell一层一层加到cell_list中
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          activation=activation[i],
                                          peephole=peephole,
                                          batchnorm=batchnorm))
        
        # nn.ModuleList 是一个专门用于包装子模块列表的 PyTorch 内置模块，
        # 使用它可以让子模块的参数被自动注册到父模块中。
        # 这里 cell_list 是由多个 ConvLSTMCell 组成的列表，使用 nn.ModuleList 
        # 包装后，cell_list 中的每个 ConvLSTMCell 都会被自动注册到 ConvLSTM 的参数中
        self.cell_list = nn.ModuleList(cell_list)

        # 表示 ConvLSTM 模型的计算设备为 CPU
        # 如果我想将模型移动到GPU上,可以怎么做?
        # 可以通过调用 to() 方法将模型转移到 GPU 上
        # model.to(device)
        self.device = 'cpu'
        # 在初始化时调用的方法，用于随机初始化模型的参数
        self.reset_parameters()

    def forward(self, input, hidden_state):
        """

        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            这里这个隐藏状态的初始大小是多少,为什么要传这个进来?
            这个隐藏状态hidden_state的初始大小是由模型的参数确定的，
            一般情况下是一个元组(num_layers, batch_size, hidden_dim, height, width)，
            其中num_layers表示模型的层数，batch_size表示输入的batch size，
            hidden_dim表示每个单元的隐藏状态的维度，height和width分别表示输入数据的高度和宽度。
            传入hidden_state的作用是让模型可以在多次调用时使用相同的初始状态，
            从而保持模型的连续性和稳定性。
            如果没有传入hidden_state，则会在内部调用get_init_states()函数生成一个新的初始状态
        Returns
        -------
        last_state_list, layer_output
        """
        # 将输入张量按指定的维度dim进行解绑定（unbind），
        # 返回一个包含张量各个部分的元组（tuple）。
        # 这里指定的维度为int(self.batch_first)，
        # 也就是在batch_first为True的情况下将第1个维度解绑，
        # 即将形状为(batch_size, seq_len, ...)的输入张量按序列长度（seq_len）
        # 拆分为长度为1的张量的序列，每个张量形状为(batch_size, ...)
        # 这样做的目的是为了方便逐个时间步地对输入进行处理，
        # 每个时间步的输入形状为(batch_size, ...)。
        # 这里是按照batch将输入分成不同批次来处理吗?
        # 问了chatgpt,似乎大概率是这样
        # int(True)值为1,int(False)值为0
        cur_layer_input = torch.unbind(input, dim=int(self.batch_first))

        # 检查是否提供了初始的隐藏状态，
        # 如果没有提供则通过 get_init_states() 方法来生成一个初始状态
        if not hidden_state:
            hidden_state = self.get_init_states(cur_layer_input[0].size(int(not self.batch_first)))

        # 获取当前输入的序列长度，即将当前输入张量按时间步拆分后得到的列表长度
        # 根据unbind操作,这里的序列长度会自适应为seq_len而不是batchsize
        seq_len = len(cur_layer_input)

        # 用于存储循环神经网络每一层的输出和最终状态。
        # 在每一次循环中，这两个列表会分别添加每一层的输出和状态
        layer_output_list = []
        last_state_list   = []

        # ConvLSTM模型的核心部分，包含了对每一层LSTM单元的处理
        for layer_idx in range(self.num_layers):
            # 从上一次的hidden_state中取出该层的hidden_state（h）和cell state（c）
            h, c = hidden_state[layer_idx]
            output_inner = []
            # 对于每个时间步（t），
            # 将当前时间步的输入（cur_layer_input[t]）
            # 和上一个时间步的hidden_state和cell state（prev_state=[h, c]）作为参数传入该层的LSTM单元，
            # 计算当前时间步的hidden_state和cell state
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input=cur_layer_input[t],
                                                 prev_state=[h, c])
                # 将所有时间步的hidden_state组成的列表output_inner保存下来，作为下一层的输入
                output_inner.append(h)

            # 将output_inner作为下一层的输入，继续执行上述步骤
            cur_layer_input = output_inner
            # 将该层的最后一个时间步的hidden_state和cell state保存到last_state_list中
            last_state_list.append((h, c))

        # 将一个列表中的张量按照指定维度拼接成一个张量。
        # 具体地，output_inner是一个张量列表，每个张量的形状是 (batch_size, hidden_dim)。
        # 使用 torch.stack 函数将这些张量沿着指定维度拼接成
        # 一个形状为 (seq_len, batch_size, hidden_dim) 的张量，
        # 其中 seq_len 是序列长度，batch_size 是批次大小，hidden_dim 是隐藏状态维度。
        # int(self.batch_first) 将 self.batch_first 的布尔值转换成整型，
        # 指定拼接的维度。
        # 如果 self.batch_first 是 True，则将张量列表沿着第二个维度拼接，
        # 即将 batch_size 这个维度排在前面；
        # 否则将沿着第一维拼接，即将 seq_len 这个维度排在前面。
        layer_output = torch.stack(output_inner, dim=int(self.batch_first))

        # layer_output: 经过所有层的ConvLSTM后，输出的所有状态经过堆叠后的结果。
        # 如果batch_first为True，
        # 则返回形状为(batch_size, seq_len, num_filters, height, width)的5D张量，
        # 否则返回形状为(seq_len, batch_size, num_filters, height, width)的5D张量
        #  
        # last_state_list：每一层最后一个时间步的隐藏状态h和细胞状态c，
        # 存储在一个元组中，并以列表的形式返回。列表的长度为num_layers，
        # 每个元素的形状为(batch_size, num_filters, height, width)
        return layer_output, last_state_list

    # 循环遍历模型中每一个 ConvLSTMCell，并调用其自身的 reset_parameters 函数。因此，
    # 可以通过调用 ConvLSTM 的 reset_parameters 函数来
    # 重新初始化整个模型中所有 ConvLSTMCell 的权重参数
    def reset_parameters(self):
        for c in self.cell_list:
            c.reset_parameters()

    # 用于返回多层LSTM的初始化隐藏状态
    def get_init_states(self, batch_size, cuda=True, device='cuda'):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, cuda, device))
        return init_states

    # 这里的静态方法和C++中的静态成员函数作用类似
    @staticmethod
    # 用于检查传入的kernel_size参数是否符合要求
    # Kernel_size必须是元组或元组列表
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list)
            and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`Kernel_size` must be tuple or list of tuples')

    @staticmethod
    # 用于将单个参数 param 扩展成一个多层网络所需的参数列表，
    # 返回一个长度为 num_layers 的列表，
    # 其中每个元素都是 param。如果 param 已经是列表，
    # 则直接返回该列表。例如，如果 param 是一个浮点数，
    # 则将其扩展为具有 num_layers 个相同值的浮点数列表。
    # 这个方法主要用于方便用户构建多层的 ConvLSTM 网络
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
