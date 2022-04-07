import torch
import torch.nn as nn
from functools import reduce


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


def load_estimation_model(inputfilename, outputfilename, begin, end, csv_export=True):
    with open(inputfilename, "r") as rf:
        contents = rf.read()
        contents = contents.split(",")

    data = torch.Tensor(1, 350)
    name = ""
    for i in range(len(contents)):
        if i == 0:
            name = contents[i].strip()
        else:
            val = float(contents[i].strip())
            data[0][i - 1] = val

    model = nn.Sequential(
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(350, 1024)),
        nn.Sigmoid(),
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(1024, 512)),
        nn.Sigmoid(),
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(512, 256)),
        nn.Sigmoid(),
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(256, 4)),
    )

    model.load_state_dict(torch.load("em.pth"))
    my_prediction = model.forward(data)

    prediction_dict = {}
    prediction_dict["F1"] = 1000 * float(my_prediction[0][0])
    prediction_dict["F2"] = 1000 * float(my_prediction[0][1])
    prediction_dict["F3"] = 1000 * float(my_prediction[0][2])
    prediction_dict["F4"] = 1000 * float(my_prediction[0][3])

    if csv_export:
        with open(outputfilename, "w") as wf:
            wf.write("NAME,begin,end,F1,F2,F3,F4\n")
            wf.write(name + "," + str(begin) + "," + str(end) + "," + \
                     str(prediction_dict["F1"]) + "," + str(prediction_dict["F2"]) + "," + \
                     str(prediction_dict["F3"]) + "," + str(prediction_dict["F4"]) + "\n")

    return prediction_dict
