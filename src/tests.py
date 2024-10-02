import os, unittest
import torch
import transformers

import utils
import ft, icl

os.environ["DEVICE"] = "cpu"
DEVICE = torch.device("cpu")


class Test_ft_params(unittest.TestCase):
    def setUp(self):
        # Cache the datasets and models needed
        utils.get_model_and_tokenizer('bert-tiny', transformers.AutoModelForCausalLM)
        utils.get_model_and_tokenizer('bert-tiny', transformers.AutoModelForSequenceClassification, num_labels=5)
        utils.get_dataset(datasetID='amazon', n_train=1, n_val=125)

    def test(self):
        """Basic test case for testing the number of parameters to fine tune in mode 'all'."""
        
        model, _ = utils.get_model_and_tokenizer('bert-tiny', transformers.AutoModelForCausalLM)
        parameters = [parameter for parameter in ft.parameters_to_fine_tune(model, "all")]
        print(len(parameters))

        # Check that the number of parameters to be optimized match
        self.assertTrue(len(parameters) == 42, "Incorrect number of parameters to be optimized returned by parameters_to_fine_tune!")

class Test_icl(unittest.TestCase):
    def setUp(self):
        # Cache the datasets and models needed
        utils.get_model_and_tokenizer('gpt2-med', transformers.AutoModelForCausalLM)

        self.get_icl_prompts = icl.get_icl_prompts
        self.do_sample = icl.do_sample

        self.prompt_ids = torch.tensor([[  464,  4141,    12, 11990,   609, 38630, 16055, 12052,    11,   636,
           286,   350,  1142,   375, 15868,   446,    11,  3382,   284,  1382,
           319,   262,  2524,   286,   262, 11773,  1233, 14920,  1474,  1879,
          1313,    11,   543,   468,   407,   587,   973,   329,  1478,   812,
            13,   383,  2656,  1233, 14920,  6832,   423,  1541,   587, 33359,
            13, 21913,   290, 11344,  2594,  5583,  8900, 15796,  9847,   531,
           262,   649,  1233, 14920,   561,   307,  7062,    13,   632,   561,
          1612,   281,  2620,   286,   838,     4,   287,   262,  1664,   338,
         26868, 39501,  1233,  4509,  5339,    13,   609, 38630,   318,   262,
          1218,  4094,  1664,   287, 46755, 39501,    11,   351,   546,  1160,
             4,   286,   262,  1910,    13, 30305,   329,   257,  1688,   649,
          1233, 14920,   287,  2531,   893,   485,   423,   587,  6325,   416,
          3461,   323, 43321,    13,   383, 35090, 19862, 10571,   357, 23055,
         19862,  4608,     8,  3952,   318,  1363,   284,   262,  4387, 10368,
           286,   262,   995,   338,  5637,   530,    12, 25311,   276,  9529,
           259,   420, 27498,    13,   198, 21300,   942,  3751,  3952,  3085,
          1866, 17247,   290,   788, 32331,   262, 31134,   832,   262,  3952,
            13,   198,  1026,   318,   407,  1900,   810,   262, 31134,   338,
          2802,   318,    11,   475, 49879,   318,   257,  2219,  1917,   287,
         16385,  1723,    64,    13,   198, 14565,  2056,  8636,   326,  1105,
          9529,   259,   420, 27498,   547,  2923,   416,   745, 17892,  1201,
          3269,   428,   614,    13]])

    def test_customPrompt_format(self):
        """Basic test case for checking the custom prompt format is different from the others."""

        support_inputs = ['Sandra travelled to the kitchen. John went to the office. Where is Sandra?']
        support_labels = ['kitchen']
        test_input = 'Sandra went to the office. Daniel went back to the garden. Where is Sandra?'

        # Get all types of prompts
        utils.fix_random_seeds()
        prompt_qa = self.get_icl_prompts(support_inputs, support_labels, test_input, prompt_mode='qa')

        utils.fix_random_seeds()
        prompt_none = self.get_icl_prompts(support_inputs, support_labels, test_input, prompt_mode='none')

        utils.fix_random_seeds()
        prompt_tldr = self.get_icl_prompts(support_inputs, support_labels, test_input, prompt_mode='tldr')

        utils.fix_random_seeds()
        prompt_custom = self.get_icl_prompts(support_inputs, support_labels, test_input, prompt_mode='custom')

        # Check that custom prompt is different than the rest of our specified formats
        self.assertTrue(
            prompt_custom != prompt_qa and prompt_custom != prompt_none and prompt_custom != prompt_tldr, 
            "Custom prompt format from get_icl_prompts needs to be different than other formats!"
        )  
        print('Custom prompt format from get_icl_prompts is good!')

    def test_do_sample(self):
        """Basic test case for checking that do_sample does not include the input_ids prefix OR the stop token."""

        model, tokenizer = utils.get_model_and_tokenizer('gpt2-med', transformers.AutoModelForCausalLM)
        stop_tokens = utils.stop_tokens(tokenizer)
        
        max_tokens = utils.max_sampled_tokens_for_dataset('xsum')
        
        sampled_tokens = self.do_sample(model=model.to(DEVICE), input_ids=self.prompt_ids.to(DEVICE), stop_tokens=stop_tokens, max_tokens=max_tokens)

        # Check that the sampled tokens do not include the input_ids prefix
        self.assertFalse(all(x in sampled_tokens for x in list(self.prompt_ids.squeeze().numpy())), "do_sample should not include the input_idx prefix!")  

        # Check that the sampled tokens do not include the stop tokens
        self.assertFalse(any(x in sampled_tokens for x in stop_tokens), "do_sample should not include any stop token!")  

        # Check that we sampled at most max_tokens tokens
        self.assertTrue(len(sampled_tokens) > 0 and len(sampled_tokens) <= max_tokens, f"do_sample should at most sample {max_tokens} tokens!")  

        print('icl.do_sample is good!')

    def test_allPrompts_format(self):
        """Basic test case for checking the prompt count and spacing."""

        support_inputs = ['Sandra travelled to the kitchen. John went to the office. Where is Sandra?',
                            'Mike worked in the garden. Terry went to the yard. Where is Mike?']
        support_labels = ['kitchen',
                            'garden']
        test_input = 'Sandra went to the office. Daniel went back to the garden. Where is Sandra?'

        for n_support in range(len(support_inputs)):
            # Get all types of prompts
            utils.fix_random_seeds()
            prompt_qa = self.get_icl_prompts(support_inputs[:n_support], support_labels[:n_support], test_input, prompt_mode='qa')
            assert prompt_qa[-1] != " ", "Prompt format for qa should not have a space at the end"
            assert prompt_qa.count(' In the') == n_support + 1, "Prompt count for qa is incorrect"

            utils.fix_random_seeds()
            prompt_none = self.get_icl_prompts(support_inputs[:n_support], support_labels[:n_support], test_input, prompt_mode='none')
            assert prompt_none[-1] != " ", "Prompt format for none should not have a space at the end"

            utils.fix_random_seeds()
            prompt_tldr = self.get_icl_prompts(support_inputs[:n_support], support_labels[:n_support], test_input, prompt_mode='tldr')
            assert prompt_tldr[-1] != " ", "Prompt format for tldr should not have a space at the end"
            assert prompt_tldr.count(' TL;DR:') == n_support + 1, "Prompt count for tldr is incorrect"

            utils.fix_random_seeds()
            prompt_custom = self.get_icl_prompts(support_inputs[:n_support], support_labels[:n_support], test_input, prompt_mode='custom')
            assert prompt_custom[-1] != " ", "Prompt format for custom should not have a space at the end"

        print('Prompt formats are all good!')
