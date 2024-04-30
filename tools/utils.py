class LoopPadding(object):
    '''
    序列填充：当序列长度小于设定的最小长度时，循环填充最后一个 index 至长度等于设定的最小长度
    '''
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

class SlidingWindow(object):
    '''
    滑动窗口：根据子序列长度和滑动步长，对一个长序列进行采样，组成多个长度相等的子序列
    '''
    def __init__(self, size, stride=0):
        self.size = size
        if stride == 0:
            self.stride = self.size
        else:
            self.stride = stride
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        out = []
        for begin_index in frame_indices[::self.stride]:
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out


