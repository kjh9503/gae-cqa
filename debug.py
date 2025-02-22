import torch
import numpy as np
import torch.nn as nn
import copy
x = torch.randn((5,3))
print(x)
x_ = torch.clone(x[[1, 3]])
print(x_)
y = torch.randn((2, 3))
print(y)
x[[1, 3]] = y
print(x)
print(x_)
exit()
x = [1,2,3,0,6,7,8,]
mask = np.random.binomial(1, 0.6, len(x))  # Bernoulli distribution samples (0 or 1) with probability p
print(mask)
sampled_lst = [x[i] for i in range(len(x)) if mask[i] == 1]
print(sampled_lst)
exit()
x = torch.FloatTensor(0)
crit = nn.MSELoss()
l = crit(torch.FloatTensor(234.2), torch.FloatTensor(222,3))
print(l.size());exit()
l += x
exit()
x = 0
# 예시 텐서
x = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
y = torch.tensor([4, 5, 6, 7, 8, 6, 7, 8, 6, 7, 8, 8])

# x 텐서의 마지막 값을 유지하기 위한 인덱스 마스크를 계산
_, last_indices = torch.unique_consecutive(x, return_inverse=True)
print(last_indices.flip(0))
exit()
_, inverse_indices = torch.unique_consecutive(last_indices.flip(0), return_inverse=True)
last_index_positions = (x.size(0) - 1) - inverse_indices.flip(0)

# y 텐서에서 해당 인덱스 값을 추출
output = y[last_index_positions]

print(output)