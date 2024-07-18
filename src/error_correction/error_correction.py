from transformers import ElectraForPreTraining, AutoTokenizer, BertForMaskedLM, BartForConditionalGeneration
import torch
import re
# import sacremoses
# detok = sacremoses.MosesDetokenizer('en')

class gen_text():
    def __init__(self, bart_pretrained_version):
        super().__init__()
        self.bart_pretrained_version = bart_pretrained_version
        self.discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator").to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bart_tokenizer = AutoTokenizer.from_pretrained(self.bart_pretrained_version)
        self.bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to("cuda")
        # self.normalizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to("cuda")
        # self.normalizer_model_trained = torch.load("normalizer_model4.pth")
        # self.normalizer_model.load_state_dict(self.normalizer_model_trained.state_dict())
        self.vocabsss = self.tokenizer.get_vocab()
        self.vocabs = dict((v,k) for k,v in self.vocabsss.items())


    def identify_fake_tokens(self, fake_inps, preds):
        for key, val in enumerate(fake_inps):
            # print(val.item())
            if preds[key] > -1 and self.vocabs[val.item()].startswith('##')==False:
                fake_inps[key] = 1
        return fake_inps


    def clean_text(self,input_text):
        input_text = ' '.join([ word if not word.startswith('[unused0]') else '<mask>' for word in input_text.split()])
        # input_text = detok.detokenize(input_text.split(" "))
        # input_text = input_text.capitalize()
        # input_text = detok.detokenize(input_text.split(" "))

        return input_text



    def generate(self, text_list, max_length, num_beams):
        self.discriminator.eval()
        # self.normalizer_model.eval()
        self.bart_model.eval()
        with torch.no_grad(): 
            fake_in = self.tokenizer.batch_encode_plus(text_list, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to("cuda")
            fake_inputs = fake_in["input_ids"]
            fake_input_attention_mask = fake_in["attention_mask"]
            discriminator_outputs = self.discriminator(fake_inputs)
            predictions = discriminator_outputs.logits
            preds = predictions.squeeze().tolist()
            output_ids = list(map(self.identify_fake_tokens, fake_inputs, preds))
            output_texts1 = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            output_texts1 = list(map(self.clean_text, output_texts1))
            inputs = self.bart_tokenizer.batch_encode_plus(output_texts1, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to("cuda")
            summary_ids = self.bart_model.generate(input_ids = inputs["input_ids"], num_beams=num_beams, do_sample=False, min_length=0, max_length=200)
            # summary_ids = self.normalizer_model.generate(input_ids = summary_ids, num_beams=1, do_sample=False, min_length=0, max_length=200)
            output_text = self.bart_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_texts1, output_text


