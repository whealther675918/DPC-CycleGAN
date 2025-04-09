import cv2
import torchvision.transforms as transforms
from .generator import Generator



image = cv2.imread('/media/whealther/diskb/UWICN-master/dataset/air_image/000002.png')
image = transforms.functional.to_tensor(image)
image = image.unsqueeze(0)

print(image.shape)

net = Generator()
image = net(image)

print('-----save------')
image.save('/home/whealther/whealther230224/result.png')

