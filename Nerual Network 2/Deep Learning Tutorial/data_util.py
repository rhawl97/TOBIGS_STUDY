import torchvision
import torchvision.transforms as transforms

# 불러온 이미지 선처리 하기 위한 단계 (ToTensor : 이미지를 텐서 형태로)
# Tensor의 range는 0에서 1로
# 채널에 변화를 줌(?)
# CenterCrop : 이미지 가운데 부분만 잘라서 사용하겠다
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
# 이미지 전처리를 위한 transforms.Compose
# PIL image (H x W x C) -> Tensor (C x H x W)

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=False,
                                        transform=transform)


testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=False,
                                       transform=transform)
