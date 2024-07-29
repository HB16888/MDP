from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import PixArtSigmaPipeline
from torch import nn, einsum
import torch
from transformers import CLIPTokenizer
from transformers import CLIPTextModel
from transformers import T5EncoderModel
from transformers import T5Tokenizer
class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="/data3/ipad_3d/HuggingFace-Download-Accelerator/models--openai--clip-vit-large-patch14", device="cuda", max_length=77, pool=True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

        self.pool = pool

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        if self.pool:
            z = outputs.pooler_output
        else:
            z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

def main():
    model="PixArtSigma"
    # 定义你需要的句子
    fixed_sentence = "A photo shows cars on city road.".lower().strip()
    if model == "CLIP":
        text_encoder = FrozenCLIPEmbedder(max_length=20)
        text_encoder.cuda()


        with torch.no_grad():
            zeroshot_weights = []
            texts = fixed_sentence # 使用固定句子
            class_embeddings = text_encoder.encode(texts).detach().mean(dim=0)
            zeroshot_weights.append(class_embeddings)
        torch.save(zeroshot_weights, f'{model}_kitti_embeddings.pth')
    elif model == "T5":
        device="cuda"
        tokenizer = T5Tokenizer.from_pretrained("/data3/ipad_3d/HuggingFace-Download-Accelerator/models--PixArt-alpha--PixArt-Sigma-XL-2-1024-MS",subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained("/data3/ipad_3d/HuggingFace-Download-Accelerator/models--PixArt-alpha--PixArt-Sigma-XL-2-1024-MS",subfolder="text_encoder").cuda().eval()

        with torch.no_grad():
            zeroshot_weights = []
            texts = fixed_sentence
            batch_encoding = tokenizer(texts, truncation=True, max_length=300, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(device)
            outputs = text_encoder(input_ids=tokens)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (tokens.to(dtype=torch.int, device=last_hidden_state.device) == 1)
                .int()
                .argmax(dim=-1),
            ]
            class_embeddings = pooled_output.detach().mean(dim=0)
            zeroshot_weights.append(class_embeddings)
        torch.save(zeroshot_weights, f'{model}_kitti_embeddings.pth')
    elif model == "PixArtSigma":
        device="cuda"
        pipe = PixArtSigmaPipeline.from_pretrained(
            "/data3/ipad_3d/HuggingFace-Download-Accelerator/models--PixArt-alpha--PixArt-Sigma-XL-2-1024-MS").to("cuda")
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask=pipe.encode_prompt(prompt=fixed_sentence,device=device)
            
            torch.save(prompt_embeds, f'{model}_kitti_prompt_embeds.pth')
            torch.save(prompt_attention_mask, f'{model}_kitti_prompt_attention_mask.pth')

if __name__ == '__main__':
    main()
