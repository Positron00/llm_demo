import os, unittest
import torch
import transformers

import utils
import ft

os.environ["DEVICE"] = "cpu"
DEVICE = torch.device("cpu")


class Test_ft_params(unittest.TestCase):
    def setUp(self):
        # Cache the datasets and models needed
        utils.get_model_and_tokenizer('bert-tiny', transformers.AutoModelForCausalLM)
        utils.get_model_and_tokenizer('bert-tiny', transformers.AutoModelForSequenceClassification, num_labels=5)
        utils.get_dataset(dataset='amazon', n_train=1, n_val=125)

    def test(self):
        """Basic test case for testing the number of parameters to fine tune in mode 'all'."""
        
        model, _ = utils.get_model_and_tokenizer('bert-tiny', transformers.AutoModelForCausalLM)
        parameters = [parameter for parameter in ft.parameters_to_fine_tune(model, "all")]
        print(len(parameters))

        # Check that the number of parameters to be optimized match
        self.assertTrue(len(parameters) == 42, "Incorrect number of parameters to be optimized returned by parameters_to_fine_tune!")
