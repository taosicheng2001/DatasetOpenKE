import openke
from openke.config import Trainer, Tester
from openke.module.model import TransH
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


# dataloader for test
test_dataloader = TestDataLoader("./DatasetOpenKE/", "link")


# define the model
transh = TransH(
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

# test the model
transh.load_checkpoint('./checkpoint/transh.ckpt')
tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)