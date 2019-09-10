import torch
import torch.utils.data as torchdata
import torch.nn as nn
import numpy as np
import tqdm
import utils
import time

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from models.model import CifarResNeXt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='R110_C10')
parser.add_argument('--data_dir', default='data/')
parser.add_argument('--load', default=None)
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--agent_state', default='finetune')
parser.add_argument('--model_dir', default='')
parser.add_argument('--output', default="output.txt")
args = parser.parse_args()

#---------------------------------------------------------------------------------------------#
class FConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FConv2d, self).forward(x)
        output_area = output.size(-1)*output.size(-2)
        filter_area = np.prod(self.kernel_size)
        self.num_ops += 2*self.in_channels*self.out_channels*filter_area*output_area
        return output

class FLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FLinear, self).__init__(in_features, out_features, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FLinear, self).forward(x)
        self.num_ops += 2*self.in_features*self.out_features
        return output

def count_flops(model, reset=True):
    op_count = 0
    for m in model.modules():
        if hasattr(m, 'num_ops'):
            op_count += m.num_ops
            if reset: # count and reset to 0
                m.num_ops = 0

    return op_count

# replace all nn.Conv and nn.Linear layers with layers that count flops
nn.Conv2d = FConv2d
nn.Linear = FLinear

#--------------------------------------------------------------------------------------------#

def test(budget_constraint):

    total_ops = []
    matches, policies = [], []
    inference_time = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        targets = targets.cuda(async=True)
        inputs = inputs.cuda()
        with torch.no_grad():
            time_st = time.time()
            budget = torch.ones(targets.shape).cuda() * budget_constraint
            probs, _ = agent(inputs, budget)

            policy = probs.clone()
            policy[policy<0.5] = 0.0
            policy[policy>=0.5] = 1.0

            preds = rnet.forward_single(inputs, policy.data.squeeze(0))
            inference_time.append((time.time()-time_st)*1000.0)
            _ , pred_idx = preds.max(1)
            match = (pred_idx==targets).data.float()

            matches.append(match)
            policies.append(policy.data)

            ops = count_flops(agent) + count_flops(rnet)
            total_ops.append(ops)

    accuracy, _, sparsity, variance, policy_set = utils.performance_stats(policies, matches, matches)
    ops_mean, ops_std = np.mean(total_ops), np.std(total_ops)
    inference_time_mean, inference_time_std = np.mean(inference_time), np.std(inference_time)
    log_str = u'''
    Accuracy: %.3f
    Block Usage: %.3f \u00B1 %.3f
    FLOPs/img: %.2E \u00B1 %.2E
    Unique Policies: %d
    Average Inference time: %.3f \u00B1 %.3f
    '''%(accuracy, sparsity, variance, ops_mean, ops_std, len(policy_set), inference_time_mean, inference_time_std)
    print("======================== budget constraint: " + str(budget_constraint) + " =========================")
    print(log_str)
    print("%.3f/%.3f/%.3f/%.2E/%.2E/%.3f/%.3f/%d"%(accuracy, sparsity, variance, ops_mean, ops_std, inference_time_mean, inference_time_std, len(policy_set)))
    with open(args.output, 'a') as f:
       f.write("%.2f,%.3f,%.3f,%.3f,%.2E,%.2E,%.3f,%.3f,%.3f,%.3f,%d\n"%(budget_constraint, accuracy, sparsity, variance, ops_mean, ops_std, ops_mean, ops_std, 
inference_time_mean,
inference_time_std, len(policy_set)))


#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
testloader = torchdata.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

num_blocks = (args.depth-2)//3 * args.cardinality
agent = utils.get_budget_constraint_agent(num_blocks)
dataset = args.model.split('_')[1]
if dataset=='C10':
    rnet = CifarResNeXt(args.cardinality, args.depth, 10, args.base_width, args.widen_factor)
elif dataset=='C100':
    rnet = CifarResNeXt(args.cardinality, args.depth, 100, args.base_width, args.widen_factor)

if args.load is not None:
    if args.agent_state == "finetune":
        checkpoint = torch.load(args.load)
        rnet.load_state_dict(checkpoint['resnet'])
        agent.load_state_dict(checkpoint['agent'])
    else:
        loaded_state_dict = torch.load(args.model_dir)
        temp = {}
        for key, val in list(loaded_state_dict.items()):
            temp[key] = val
        loaded_state_dict = temp
        rnet.load_state_dict(loaded_state_dict)
        checkpoint = torch.load(args.load)
        agent.load_state_dict(checkpoint['agent'])


rnet.eval().cuda()
agent.eval().cuda()
budget_list = [0.1+ 0.05 * i for i in range(19)]
with open(args.output, 'a') as f:
    f.write("budget, accuracy, sparsity, variance, ops_mean, ops_std, ops_mean_raw, ops_std_raw, inference_time_mean, inference_time_std, policy\n")
for budget in budget_list:
    test(budget)

