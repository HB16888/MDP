from torch import nn, einsum
import torch
from transformers import CLIPTokenizer
from transformers import CLIPTextModel

class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="/mnt/nodestor/MDP/models--openai--clip-vit-large-patch14", device="cuda", max_length=77, pool=True):
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
    # 定义你需要的句子
    fixed_sentence = "A photo shows cars on city road."


    text_encoder = FrozenCLIPEmbedder(max_length=20)
    text_encoder.cuda()


    with torch.no_grad():
        zeroshot_weights = []
        texts = fixed_sentence # 使用固定句子
        class_embeddings = text_encoder.encode(texts).detach().mean(dim=0)
        zeroshot_weights.append(class_embeddings)

    torch.save(zeroshot_weights, 'kitti_embeddings.pth')

if __name__ == '__main__':
    main()
