# --coding:utf-8 --
import torch
from torch.autograd import Variable
from torchvision import transforms
from load_img import load_img
from run_code import run_style_transfer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
style_img = load_img('pic/style1.png')
style_img = Variable(style_img).to(device)
content_img = load_img('pic/MkqNLrdXvj.png')
content_img = Variable(content_img).to(device)
input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=200)
save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
save_pic.save('pic/result2.jpg')
save_pic.show()
